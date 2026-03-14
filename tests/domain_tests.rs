use chrono::NaiveDate;
use corduroy::domain::*;

#[test]
fn latitude_valid() {
    assert!(Latitude::new(0.0).is_ok());
    assert!(Latitude::new(90.0).is_ok());
    assert!(Latitude::new(-90.0).is_ok());
    assert!(Latitude::new(45.5).is_ok());
}

#[test]
fn latitude_invalid() {
    assert!(Latitude::new(90.1).is_err());
    assert!(Latitude::new(-90.1).is_err());
    assert!(Latitude::new(180.0).is_err());
    assert!(Latitude::new(f64::NAN).is_err());
}

#[test]
fn longitude_valid() {
    assert!(Longitude::new(0.0).is_ok());
    assert!(Longitude::new(180.0).is_ok());
    assert!(Longitude::new(-180.0).is_ok());
    assert!(Longitude::new(359.75).is_ok());
    assert!(Longitude::new(270.0).is_ok()); // ERA5 convention
}

#[test]
fn longitude_invalid() {
    assert!(Longitude::new(360.1).is_err());
    assert!(Longitude::new(-180.1).is_err());
}

#[test]
fn lat_to_index_north_pole() {
    assert_eq!(Latitude::new(90.0).unwrap().to_index(), 0);
}

#[test]
fn lat_to_index_south_pole() {
    assert_eq!(Latitude::new(-90.0).unwrap().to_index(), 720);
}

#[test]
fn lat_to_index_equator() {
    assert_eq!(Latitude::new(0.0).unwrap().to_index(), 360);
}

#[test]
fn lat_to_index_mid_latitude() {
    // 45°N should be at index (90 - 45) / 0.25 = 180
    assert_eq!(Latitude::new(45.0).unwrap().to_index(), 180);
}

#[test]
fn lon_to_index_prime_meridian() {
    assert_eq!(Longitude::new(0.0).unwrap().to_index(), 0);
}

#[test]
fn lon_to_index_dateline() {
    // 180°E should be at index 180 / 0.25 = 720
    assert_eq!(Longitude::new(180.0).unwrap().to_index(), 720);
}

#[test]
fn lon_to_index_negative_convention() {
    // -90°W = 270°E, index = 270 / 0.25 = 1080
    let from_negative = Longitude::new(-90.0).unwrap().to_index();
    let from_positive = Longitude::new(270.0).unwrap().to_index();
    assert_eq!(from_negative, from_positive);
    assert_eq!(from_negative, 1080);
}

#[test]
fn bbox_valid() {
    let bbox = BoundingBox::new(35.0, 45.0, -90.0, -75.0);
    assert!(bbox.is_ok());
}

#[test]
fn bbox_lat_min_exceeds_max() {
    let bbox = BoundingBox::new(50.0, 30.0, -90.0, -75.0);
    assert!(bbox.is_err());
}

#[test]
fn bbox_lat_equal() {
    let bbox = BoundingBox::new(45.0, 45.0, -90.0, -75.0);
    assert!(bbox.is_err());
}

#[test]
fn bbox_index_ranges() {
    let bbox = BoundingBox::new(35.0, 45.0, -90.0, -75.0).unwrap();

    let lat_range = bbox.lat_index_range();
    // 45°N -> index 180, 35°N -> index 220
    assert_eq!(lat_range.start, 180);
    assert_eq!(lat_range.end, 221); // exclusive

    let lon_range = bbox.lon_index_range();
    // -90° = 270° -> index 1080, -75° = 285° -> index 1140
    assert_eq!(lon_range.start, 1080);
    assert_eq!(lon_range.end, 1141); // exclusive
}

#[test]
fn bbox_lat_values_count() {
    let bbox = BoundingBox::new(35.0, 45.0, -90.0, -75.0).unwrap();
    let lats = bbox.lat_values();
    let range = bbox.lat_index_range();
    assert_eq!(lats.len(), (range.end - range.start) as usize);
    // First value should be the northern boundary (45°N)
    assert!((lats[0] - 45.0).abs() < 1e-10);
    // Last value should be near the southern boundary (35°N)
    assert!((lats[lats.len() - 1] - 35.0).abs() < 1e-10);
}

#[test]
fn time_index_known_date() {
    // 1940-01-02T00:00:00 = 24 hours after epoch
    let dt = NaiveDate::from_ymd_opt(1940, 1, 2)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    assert_eq!(datetime_to_time_index(dt).unwrap(), 24);
}

#[test]
fn time_index_epoch() {
    let dt = NaiveDate::from_ymd_opt(1940, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    assert_eq!(datetime_to_time_index(dt).unwrap(), 0);
}

#[test]
fn time_index_before_epoch() {
    let dt = NaiveDate::from_ymd_opt(1939, 12, 31)
        .unwrap()
        .and_hms_opt(23, 0, 0)
        .unwrap();
    assert!(datetime_to_time_index(dt).is_err());
}

#[test]
fn time_index_2023() {
    // 2023-01-01T00:00:00 should be a large but valid index
    let dt = NaiveDate::from_ymd_opt(2023, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let idx = datetime_to_time_index(dt).unwrap();
    // 83 years * 365.25 * 24 ≈ 727,674 hours
    assert!(idx > 700_000);
    assert!(idx < 800_000);
}

#[test]
fn precipitation_field_value_range() {
    let data = ndarray::array![[0.001, 0.005], [0.0, 0.010]];
    let field = PrecipitationField {
        data,
        latitudes: vec![45.0, 44.75],
        longitudes: vec![270.0, 270.25],
        datetime: NaiveDate::from_ymd_opt(2023, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
    };
    let (min, max) = field.value_range();
    assert!((min - 0.0).abs() < 1e-9);
    assert!((max - 0.010).abs() < 1e-9);
}

#[test]
fn precipitation_field_value_range_with_nan() {
    let data = ndarray::array![[f32::NAN, 0.005], [0.001, f32::NAN]];
    let field = PrecipitationField {
        data,
        latitudes: vec![45.0, 44.75],
        longitudes: vec![270.0, 270.25],
        datetime: NaiveDate::from_ymd_opt(2023, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap(),
    };
    let (min, max) = field.value_range();
    assert!((min - 0.001).abs() < 1e-9);
    assert!((max - 0.005).abs() < 1e-9);
}
