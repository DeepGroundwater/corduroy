use std::path::Path;

use chrono::NaiveDate;
use corduroy::domain::PrecipitationField;
use corduroy::plot;
use ndarray::Array2;

fn make_field(data: Array2<f32>) -> PrecipitationField {
    let (n_lat, n_lon) = data.dim();
    PrecipitationField {
        data,
        latitudes: (0..n_lat).map(|i| 45.0 - i as f64 * 0.25).collect(),
        longitudes: (0..n_lon).map(|i| 270.0 + i as f64 * 0.25).collect(),
        datetime: NaiveDate::from_ymd_opt(2023, 6, 15)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap(),
    }
}

#[test]
fn render_synthetic_data() {
    let mut data = Array2::<f32>::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
            data[[i, j]] = (i * 10 + j) as f32 * 0.001;
        }
    }

    let field = make_field(data);
    let path = std::env::temp_dir().join("corduroy_test_synthetic.png");
    plot::render_heatmap(&field, &path).unwrap();

    assert!(path.exists());
    let img = image::open(&path).unwrap();
    // Should be upscaled since 10 < MIN_PIXELS (400)
    assert!(img.width() >= 400);
    assert!(img.height() >= 400);

    std::fs::remove_file(&path).ok();
}

#[test]
fn render_zero_precip() {
    let data = Array2::<f32>::zeros((5, 5));
    let field = make_field(data);
    let path = std::env::temp_dir().join("corduroy_test_zero.png");

    // Should not panic on all-zero data
    plot::render_heatmap(&field, &path).unwrap();
    assert!(path.exists());

    std::fs::remove_file(&path).ok();
}

#[test]
fn render_nan_handling() {
    let mut data = Array2::<f32>::zeros((5, 5));
    data[[0, 0]] = f32::NAN;
    data[[2, 3]] = f32::NAN;
    data[[4, 4]] = 0.01;

    let field = make_field(data);
    let path = std::env::temp_dir().join("corduroy_test_nan.png");

    // Should not panic with NaN values
    plot::render_heatmap(&field, &path).unwrap();
    assert!(path.exists());

    std::fs::remove_file(&path).ok();
}

#[test]
fn render_empty_field_returns_error() {
    let data = Array2::<f32>::zeros((0, 0));
    let field = make_field(data);
    let path = Path::new("/tmp/corduroy_test_empty.png");

    assert!(plot::render_heatmap(&field, path).is_err());
}

#[test]
fn render_single_cell() {
    let data = Array2::from_elem((1, 1), 0.005_f32);
    let field = make_field(data);
    let path = std::env::temp_dir().join("corduroy_test_single.png");

    plot::render_heatmap(&field, &path).unwrap();
    assert!(path.exists());

    std::fs::remove_file(&path).ok();
}
