use std::path::PathBuf;

use chrono::NaiveDateTime;
use clap::Parser;

use corduroy::data;
use corduroy::domain::BoundingBox;
use corduroy::plot;

#[derive(Parser)]
#[command(name = "corduroy", about = "ERA5 precipitation mapping")]
struct Cli {
    /// Bounding box as "lat_min,lat_max,lon_min,lon_max"
    #[arg(long, value_parser = parse_bbox)]
    bbox: BoundingBox,

    /// UTC datetime in ISO 8601 format (e.g. "2023-06-15T12:00:00")
    #[arg(long)]
    datetime: String,

    /// Output PNG file path
    #[arg(short, long, default_value = "precip.png")]
    output: PathBuf,
}

fn parse_bbox(s: &str) -> Result<BoundingBox, String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err("expected 4 comma-separated values: lat_min,lat_max,lon_min,lon_max".into());
    }
    let vals: Vec<f64> = parts
        .iter()
        .map(|p| p.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("invalid number: {e}"))?;

    BoundingBox::new(vals[0], vals[1], vals[2], vals[3]).map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Parse datetime
    let datetime = NaiveDateTime::parse_from_str(&cli.datetime, "%Y-%m-%dT%H:%M:%S")
        .map_err(|e| format!("invalid datetime '{}': {e}", cli.datetime))?;

    tracing::info!(
        "Fetching precipitation for bbox [{}, {}] x [{}, {}] at {}",
        cli.bbox.lat_min.value(),
        cli.bbox.lat_max.value(),
        cli.bbox.lon_min.value(),
        cli.bbox.lon_max.value(),
        datetime,
    );

    // Create store and fetch data
    let store = data::create_store().await?;
    let field = data::fetch_precipitation(store, &cli.bbox, datetime).await?;

    let (min_val, max_val) = field.value_range();
    println!(
        "Fetched {} x {} grid points",
        field.data.nrows(),
        field.data.ncols()
    );
    println!(
        "Precipitation range: {:.4} – {:.4} mm",
        min_val * 1000.0,
        max_val * 1000.0
    );

    // Render heatmap
    plot::render_heatmap(&field, &cli.output)?;
    println!("Wrote {}", cli.output.display());

    Ok(())
}
