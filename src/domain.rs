use chrono::NaiveDateTime;
use ndarray::Array2;

/// ERA5 grid constants
pub const LAT_COUNT: usize = 721;
pub const LON_COUNT: usize = 1440;
pub const GRID_STEP: f64 = 0.25;
pub const LAT_START: f64 = 90.0; // 90°N at index 0

/// ERA5 37 standard pressure levels (hPa), top-of-atmosphere to surface.
/// Index 0 = 1 hPa (TOA), index 36 = 1000 hPa (surface).
pub const ERA5_PRESSURE_LEVELS: [f64; 37] = [
    1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 70.0,
    100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 300.0, 350.0, 400.0,
    450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 775.0, 800.0, 825.0,
    850.0, 875.0, 900.0, 925.0, 950.0, 975.0, 1000.0,
];

/// ERA5 time epoch: 1940-01-01T00:00:00 UTC
fn era5_epoch() -> NaiveDateTime {
    chrono::NaiveDate::from_ymd_opt(1940, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
}

#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("latitude {0} out of range [-90, 90]")]
    InvalidLatitude(f64),
    #[error("longitude {0} out of range [-180, 360]")]
    InvalidLongitude(f64),
    #[error("lat_min ({0}) must be less than lat_max ({1})")]
    InvalidLatRange(f64, f64),
    #[error("lon_min ({0}) must be less than lon_max ({1}) after normalization")]
    InvalidLonRange(f64, f64),
    #[error("datetime {0} is before the ERA5 epoch (1940-01-01)")]
    DatetimeBeforeEpoch(NaiveDateTime),
    #[error("time index {index} exceeds array dimension length {dim_len}")]
    TimeIndexOutOfBounds { index: u64, dim_len: u64 },
    #[error("antimeridian-crossing bounding boxes are not yet supported")]
    AntimeridianCrossing,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Latitude(f64);

impl Latitude {
    pub fn new(v: f64) -> Result<Self, DomainError> {
        if !(-90.0..=90.0).contains(&v) {
            return Err(DomainError::InvalidLatitude(v));
        }
        Ok(Self(v))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    /// Convert latitude to ERA5 grid index.
    /// Latitude runs from 90°N (index 0) to 90°S (index 720).
    pub fn to_index(self) -> usize {
        let idx = ((LAT_START - self.0) / GRID_STEP).round() as isize;
        idx.clamp(0, (LAT_COUNT - 1) as isize) as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Longitude(f64);

impl Longitude {
    pub fn new(v: f64) -> Result<Self, DomainError> {
        if !(-180.0..=360.0).contains(&v) {
            return Err(DomainError::InvalidLongitude(v));
        }
        Ok(Self(v))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    /// Normalize longitude to [0, 360) range (ERA5 convention).
    pub fn normalized(self) -> f64 {
        let v = self.0 % 360.0;
        if v < 0.0 { v + 360.0 } else { v }
    }

    /// Convert longitude to ERA5 grid index.
    /// Longitude runs from 0°E (index 0) to 359.75°E (index 1439).
    pub fn to_index(self) -> usize {
        let idx = (self.normalized() / GRID_STEP).round() as isize;
        idx.clamp(0, (LON_COUNT - 1) as isize) as usize
    }
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub lat_min: Latitude,
    pub lat_max: Latitude,
    pub lon_min: Longitude,
    pub lon_max: Longitude,
}

impl BoundingBox {
    pub fn new(
        lat_min: f64,
        lat_max: f64,
        lon_min: f64,
        lon_max: f64,
    ) -> Result<Self, DomainError> {
        let lat_min = Latitude::new(lat_min)?;
        let lat_max = Latitude::new(lat_max)?;
        let lon_min = Longitude::new(lon_min)?;
        let lon_max = Longitude::new(lon_max)?;

        if lat_min.value() >= lat_max.value() {
            return Err(DomainError::InvalidLatRange(
                lat_min.value(),
                lat_max.value(),
            ));
        }

        // Check for antimeridian crossing after normalization
        if lon_min.normalized() >= lon_max.normalized() {
            return Err(DomainError::AntimeridianCrossing);
        }

        Ok(Self {
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        })
    }

    /// Latitude index range for Zarr subsetting.
    /// Note: lat_max maps to a *smaller* index (north-to-south ordering).
    pub fn lat_index_range(&self) -> std::ops::Range<u64> {
        let start = self.lat_max.to_index() as u64; // northern boundary = smaller index
        let end = self.lat_min.to_index() as u64 + 1; // southern boundary = larger index
        start..end
    }

    /// Longitude index range for Zarr subsetting.
    pub fn lon_index_range(&self) -> std::ops::Range<u64> {
        let start = self.lon_min.to_index() as u64;
        let end = self.lon_max.to_index() as u64 + 1;
        start..end
    }

    /// Latitude values for each grid point in the subset.
    pub fn lat_values(&self) -> Vec<f64> {
        let range = self.lat_index_range();
        (range.start..range.end)
            .map(|i| LAT_START - (i as f64) * GRID_STEP)
            .collect()
    }

    /// Longitude values for each grid point in the subset.
    pub fn lon_values(&self) -> Vec<f64> {
        let range = self.lon_index_range();
        (range.start..range.end)
            .map(|i| (i as f64) * GRID_STEP)
            .collect()
    }
}

/// Convert a datetime to an ERA5 time index (hours since 1940-01-01).
pub fn datetime_to_time_index(dt: NaiveDateTime) -> Result<u64, DomainError> {
    let duration = dt.signed_duration_since(era5_epoch());
    let hours = duration.num_hours();
    if hours < 0 {
        return Err(DomainError::DatetimeBeforeEpoch(dt));
    }
    Ok(hours as u64)
}

/// Full-globe latitude values (721 points, 90°N → 90°S at 0.25° spacing).
pub fn era5_latitudes() -> Vec<f64> {
    (0..LAT_COUNT).map(|i| LAT_START - i as f64 * GRID_STEP).collect()
}

/// Full-globe longitude values (1440 points, 0°E → 359.75°E at 0.25° spacing).
pub fn era5_longitudes() -> Vec<f64> {
    (0..LON_COUNT).map(|i| i as f64 * GRID_STEP).collect()
}

/// Convert a time index (hours since 1940-01-01) back to a NaiveDateTime.
pub fn time_index_to_datetime(index: u64) -> NaiveDateTime {
    era5_epoch() + chrono::Duration::hours(index as i64)
}

/// A fetched precipitation field with coordinate metadata.
#[derive(Debug, Clone)]
pub struct PrecipitationField {
    pub data: Array2<f32>,
    pub latitudes: Vec<f64>,
    pub longitudes: Vec<f64>,
    pub datetime: NaiveDateTime,
}

impl PrecipitationField {
    /// Precipitation min/max, ignoring NaN values.
    pub fn value_range(&self) -> (f32, f32) {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &v in self.data.iter() {
            if v.is_nan() {
                continue;
            }
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        if min == f32::INFINITY {
            (0.0, 0.0)
        } else {
            (min, max)
        }
    }
}
