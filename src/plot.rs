use std::path::Path;

use colorgrad::Gradient;
use image::{ImageBuffer, Rgba, RgbaImage};

use crate::domain::PrecipitationField;

#[derive(Debug, thiserror::Error)]
pub enum PlotError {
    #[error("failed to save image: {0}")]
    ImageSave(#[from] image::ImageError),
    #[error("empty precipitation field")]
    EmptyField,
}

/// Minimum pixel size for the shortest image dimension.
const MIN_PIXELS: u32 = 400;

/// Width of the colorbar strip in pixels (before scaling).
const COLORBAR_WIDTH: u32 = 20;

/// Gap between heatmap and colorbar in pixels (before scaling).
const COLORBAR_GAP: u32 = 5;

/// Render a precipitation field as a PNG heatmap with a colorbar.
///
/// Precipitation values are mapped through a turbo colormap. NaN values
/// are rendered as transparent. The image is upscaled if the grid is
/// smaller than `MIN_PIXELS` on its shortest side.
pub fn render_heatmap(field: &PrecipitationField, output_path: &Path) -> Result<(), PlotError> {
    let (n_lat, n_lon) = field.data.dim();
    if n_lat == 0 || n_lon == 0 {
        return Err(PlotError::EmptyField);
    }

    let (min_val, max_val) = field.value_range();
    // Convert from meters to mm for display
    let min_mm = min_val * 1000.0;
    let max_mm = max_val * 1000.0;
    let range = if (max_mm - min_mm).abs() < 1e-9 {
        1.0 // avoid division by zero for uniform fields
    } else {
        max_mm - min_mm
    };

    // Compute scale factor for small grids
    let shortest = n_lat.min(n_lon) as u32;
    let scale = if shortest < MIN_PIXELS {
        (MIN_PIXELS + shortest - 1) / shortest // ceiling division
    } else {
        1
    };

    let heatmap_w = (n_lon as u32) * scale;
    let heatmap_h = (n_lat as u32) * scale;
    let total_w = heatmap_w + (COLORBAR_GAP + COLORBAR_WIDTH) * scale;
    let total_h = heatmap_h;

    let gradient = colorgrad::preset::turbo();
    let mut img: RgbaImage = ImageBuffer::new(total_w, total_h);

    // Render heatmap
    for lat_idx in 0..n_lat {
        for lon_idx in 0..n_lon {
            let v = field.data[[lat_idx, lon_idx]];
            let color = if v.is_nan() {
                Rgba([0, 0, 0, 0]) // transparent
            } else {
                let v_mm = v * 1000.0;
                let t = ((v_mm - min_mm) / range).clamp(0.0, 1.0);
                let [r, g, b, a] = gradient.at(t).to_rgba8();
                Rgba([r, g, b, a])
            };

            // Fill scaled pixel block
            for dy in 0..scale {
                for dx in 0..scale {
                    let px = (lon_idx as u32) * scale + dx;
                    let py = (lat_idx as u32) * scale + dy;
                    img.put_pixel(px, py, color);
                }
            }
        }
    }

    // Render colorbar
    let cb_x_start = heatmap_w + COLORBAR_GAP * scale;
    let cb_width = COLORBAR_WIDTH * scale;
    for y in 0..total_h {
        // Map from bottom (min) to top (max)
        let t = 1.0 - (y as f32 / (total_h - 1).max(1) as f32);
        let [r, g, b, a] = gradient.at(t).to_rgba8();
        let color = Rgba([r, g, b, a]);
        for x in cb_x_start..cb_x_start + cb_width {
            img.put_pixel(x, y, color);
        }
    }

    img.save(output_path)?;

    tracing::info!(
        "Precipitation range: {:.4} – {:.4} mm",
        min_mm,
        max_mm
    );

    Ok(())
}
