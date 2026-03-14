use std::collections::HashMap;
use std::sync::Arc;

use chrono::NaiveDateTime;
use ndarray::{s, Array2, Array3, Array4, Axis};
use object_store::gcp::GoogleCloudStorage;
use object_store::prefix::PrefixStore;
use tokio::task::JoinSet;
use zarrs::array::Array;
use zarrs_object_store::AsyncObjectStore;

use crate::domain::{
    datetime_to_time_index, era5_latitudes, era5_longitudes, time_index_to_datetime, BoundingBox,
    DomainError, PrecipitationField, ERA5_PRESSURE_LEVELS,
};

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("failed to build GCS store: {0}")]
    StoreCreation(#[from] object_store::Error),
    #[error("failed to open zarr array '{path}': {source}")]
    ArrayOpen {
        path: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("failed to retrieve array data: {0}")]
    ArrayRead(Box<dyn std::error::Error + Send + Sync>),
    #[error("domain validation error: {0}")]
    Domain(#[from] DomainError),
    #[error("array reshape failed: {0}")]
    Reshape(#[from] ndarray::ShapeError),
    #[error("variable '{variable}' has {ndim} dimensions (expected 3 or 4)")]
    UnexpectedDimensions { variable: String, ndim: usize },
    #[error("task join error: {0}")]
    TaskJoin(String),
    #[error("time_step_hours must be > 0")]
    InvalidTimeStep,
}

/// Concrete store type: AsyncObjectStore wrapping a PrefixStore over GCS.
pub type Era5Store = AsyncObjectStore<PrefixStore<GoogleCloudStorage>>;

/// The Zarr v3 array path for total precipitation within the ARCO ERA5 store.
/// If this is wrong, `fetch_precipitation` will fail with `DataError::ArrayOpen`.
/// Inspect the bucket to find the correct path:
///   gsutil ls gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/
const PRECIP_ARRAY_PATH: &str = "/total_precipitation";

/// Create an async readable store pointing at the ARCO ERA5 Zarr v3 dataset on GCS.
///
/// This requires no credentials — the bucket is publicly accessible.
pub async fn create_store() -> Result<Arc<Era5Store>, DataError> {
    use object_store::gcp::GoogleCloudStorageBuilder;

    let gcs = GoogleCloudStorageBuilder::new()
        .with_bucket_name("gcp-public-data-arco-era5")
        .with_skip_signature(true)
        .build()?;

    let prefix_store = PrefixStore::new(gcs, "ar/full_37-1h-0p25deg-chunk-1.zarr-v3");

    Ok(Arc::new(AsyncObjectStore::new(prefix_store)))
}

/// Fetch ERA5 total precipitation for a bounding box at a specific datetime.
///
/// Returns a `PrecipitationField` containing the 2D precipitation array (lat x lon),
/// coordinate vectors, and the requested datetime.
pub async fn fetch_precipitation(
    store: Arc<Era5Store>,
    bbox: &BoundingBox,
    datetime: NaiveDateTime,
) -> Result<PrecipitationField, DataError> {
    // Open the total_precipitation array
    tracing::info!("Opening Zarr v3 array at '{}'", PRECIP_ARRAY_PATH);
    let array = Array::async_open(store, PRECIP_ARRAY_PATH)
        .await
        .map_err(|e| DataError::ArrayOpen {
            path: PRECIP_ARRAY_PATH.to_string(),
            source: Box::new(e),
        })?;

    let shape = array.shape();
    tracing::info!(
        "Array shape: {:?}, dtype: {:?}",
        shape,
        array.data_type()
    );

    // Compute time index
    let time_idx = datetime_to_time_index(datetime)?;
    if time_idx >= shape[0] {
        return Err(DataError::Domain(DomainError::TimeIndexOutOfBounds {
            index: time_idx,
            dim_len: shape[0],
        }));
    }
    tracing::info!("Time index {} for datetime {}", time_idx, datetime);

    // Compute spatial index ranges
    let lat_range = bbox.lat_index_range();
    let lon_range = bbox.lon_index_range();
    tracing::info!(
        "Lat indices: {:?}, Lon indices: {:?}",
        lat_range,
        lon_range
    );

    let n_lat = (lat_range.end - lat_range.start) as usize;
    let n_lon = (lon_range.end - lon_range.start) as usize;

    // Retrieve the subset: [time_idx..time_idx+1, lat_range, lon_range]
    tracing::info!("Fetching {} x {} grid points...", n_lat, n_lon);
    let subset = zarrs::array::ArraySubset::new_with_ranges(&[
        time_idx..time_idx + 1,
        lat_range.start..lat_range.end,
        lon_range.start..lon_range.end,
    ]);

    let data: ndarray::ArrayD<f32> = array
        .async_retrieve_array_subset(&subset)
        .await
        .map_err(|e| DataError::ArrayRead(Box::new(e)))?;

    // Reshape from 3D (1, n_lat, n_lon) to 2D (n_lat, n_lon)
    let data_2d: Array2<f32> = data.into_shape_with_order((n_lat, n_lon))?;

    Ok(PrecipitationField {
        data: data_2d,
        latitudes: bbox.lat_values(),
        longitudes: bbox.lon_values(),
        datetime,
    })
}

// ---------------------------------------------------------------------------
// Multi-variable global ERA5 fetch (for NeuralGCM initial conditions)
// ---------------------------------------------------------------------------

/// An ERA5 variable array — either pressure-level (4D) or surface (3D).
pub enum Era5Array {
    /// Shape: (time, level, latitude, longitude)
    PressureLevel(Array4<f32>),
    /// Shape: (time, latitude, longitude)
    Surface(Array3<f32>),
}

/// A multi-variable ERA5 dataset covering a time range at full globe resolution.
///
/// Designed for feeding NeuralGCM initial conditions: pressure-level variables
/// (temperature, wind, humidity, geopotential, cloud water) and surface forcings
/// (SST, sea ice).
pub struct Era5Dataset {
    pub variables: HashMap<String, Era5Array>,
    pub times: Vec<NaiveDateTime>,
    pub latitudes: Vec<f64>,
    pub longitudes: Vec<f64>,
    pub levels: Vec<f64>,
}

/// Fetch multiple ERA5 variables at full globe resolution for a time range.
///
/// Variables are fetched in parallel (one tokio task per variable).
/// Each variable is auto-detected as pressure-level (4D) or surface (3D)
/// based on its Zarr array dimensionality.
///
/// # Arguments
/// - `variables`: Zarr array names (e.g., `["temperature", "sea_surface_temperature"]`)
/// - `time_start`, `time_end`: time range (inclusive)
/// - `time_step_hours`: step between timesteps (e.g., 6 for every 6 hours)
pub async fn fetch_era5_global(
    store: Arc<Era5Store>,
    variables: &[String],
    time_start: NaiveDateTime,
    time_end: NaiveDateTime,
    time_step_hours: u64,
) -> Result<Era5Dataset, DataError> {
    if time_step_hours == 0 {
        return Err(DataError::InvalidTimeStep);
    }

    let start_idx = datetime_to_time_index(time_start)?;
    let end_idx = datetime_to_time_index(time_end)?;

    let time_indices: Vec<u64> = (start_idx..=end_idx)
        .step_by(time_step_hours as usize)
        .collect();

    let times: Vec<NaiveDateTime> = time_indices.iter().map(|&i| time_index_to_datetime(i)).collect();

    tracing::info!(
        "Fetching {} variables for {} timesteps ({} to {}, step {}h)",
        variables.len(),
        times.len(),
        time_start,
        time_end,
        time_step_hours,
    );

    // Spawn one task per variable for parallel fetching
    let mut tasks: JoinSet<Result<(String, Era5Array), DataError>> = JoinSet::new();

    for var_name in variables {
        let store = store.clone();
        let var_name = var_name.clone();
        let time_indices = time_indices.clone();

        tasks.spawn(async move {
            fetch_single_variable(store, &var_name, &time_indices).await
        });
    }

    // Collect results
    let mut vars = HashMap::new();
    while let Some(result) = tasks.join_next().await {
        let (name, array) = result
            .map_err(|e| DataError::TaskJoin(e.to_string()))??;
        vars.insert(name, array);
    }

    Ok(Era5Dataset {
        variables: vars,
        times,
        latitudes: era5_latitudes(),
        longitudes: era5_longitudes(),
        levels: ERA5_PRESSURE_LEVELS.to_vec(),
    })
}

/// Fetch a single ERA5 variable across all requested timesteps.
///
/// Auto-detects whether the variable is pressure-level (4D) or surface (3D)
/// from the Zarr array shape.
async fn fetch_single_variable(
    store: Arc<Era5Store>,
    var_name: &str,
    time_indices: &[u64],
) -> Result<(String, Era5Array), DataError> {
    let path = format!("/{}", var_name);
    tracing::info!("Opening array '{}'", path);

    let array = Array::async_open(store, &path)
        .await
        .map_err(|e| DataError::ArrayOpen {
            path: path.clone(),
            source: Box::new(e),
        })?;

    let shape = array.shape();
    let ndim = shape.len();
    let n_times = time_indices.len();

    tracing::info!(
        "{}: shape={:?}, ndim={}, fetching {} timesteps",
        var_name,
        shape,
        ndim,
        n_times
    );

    match ndim {
        4 => {
            // Pressure-level variable: (time, level, lat, lon)
            let n_levels = shape[1] as usize;
            let n_lat = shape[2] as usize;
            let n_lon = shape[3] as usize;

            let mut result = Array4::<f32>::zeros((n_times, n_levels, n_lat, n_lon));

            for (t, &time_idx) in time_indices.iter().enumerate() {
                tracing::debug!(
                    "{}: timestep {}/{} (index {})",
                    var_name,
                    t + 1,
                    n_times,
                    time_idx
                );

                let subset = zarrs::array::ArraySubset::new_with_ranges(&[
                    time_idx..time_idx + 1,
                    0..n_levels as u64,
                    0..n_lat as u64,
                    0..n_lon as u64,
                ]);

                let chunk: ndarray::ArrayD<f32> = array
                    .async_retrieve_array_subset(&subset)
                    .await
                    .map_err(|e| DataError::ArrayRead(Box::new(e)))?;

                let chunk_4d: Array4<f32> = chunk.into_dimensionality()?;
                result
                    .slice_mut(s![t, .., .., ..])
                    .assign(&chunk_4d.index_axis(Axis(0), 0));
            }

            tracing::info!("{}: done, shape={:?}", var_name, result.dim());
            Ok((var_name.to_string(), Era5Array::PressureLevel(result)))
        }
        3 => {
            // Surface variable: (time, lat, lon)
            let n_lat = shape[1] as usize;
            let n_lon = shape[2] as usize;

            let mut result = Array3::<f32>::zeros((n_times, n_lat, n_lon));

            for (t, &time_idx) in time_indices.iter().enumerate() {
                tracing::debug!(
                    "{}: timestep {}/{} (index {})",
                    var_name,
                    t + 1,
                    n_times,
                    time_idx
                );

                let subset = zarrs::array::ArraySubset::new_with_ranges(&[
                    time_idx..time_idx + 1,
                    0..n_lat as u64,
                    0..n_lon as u64,
                ]);

                let chunk: ndarray::ArrayD<f32> = array
                    .async_retrieve_array_subset(&subset)
                    .await
                    .map_err(|e| DataError::ArrayRead(Box::new(e)))?;

                let chunk_3d: Array3<f32> = chunk.into_dimensionality()?;
                result
                    .slice_mut(s![t, .., ..])
                    .assign(&chunk_3d.index_axis(Axis(0), 0));
            }

            tracing::info!("{}: done, shape={:?}", var_name, result.dim());
            Ok((var_name.to_string(), Era5Array::Surface(result)))
        }
        _ => Err(DataError::UnexpectedDimensions {
            variable: var_name.to_string(),
            ndim,
        }),
    }
}
