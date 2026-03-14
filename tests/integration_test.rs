use chrono::NaiveDate;
use corduroy::data;
use corduroy::domain::BoundingBox;

#[tokio::test]
#[ignore] // requires network access to GCS
async fn fetch_known_bbox() {
    // Small bbox over central US — a region that frequently has precipitation
    let bbox = BoundingBox::new(38.0, 40.0, -90.0, -88.0).unwrap();
    let datetime = NaiveDate::from_ymd_opt(2023, 6, 15)
        .unwrap()
        .and_hms_opt(12, 0, 0)
        .unwrap();

    let store = data::create_store().await.expect("failed to create store");
    let field = data::fetch_precipitation(store, &bbox, datetime)
        .await
        .expect("failed to fetch precipitation");

    // 2 degrees of lat at 0.25° = 9 points, 2 degrees of lon = 9 points
    assert_eq!(field.data.nrows(), 9);
    assert_eq!(field.data.ncols(), 9);
    assert_eq!(field.latitudes.len(), 9);
    assert_eq!(field.longitudes.len(), 9);

    // Values should be non-negative (precipitation can't be negative)
    for &v in field.data.iter() {
        if !v.is_nan() {
            assert!(v >= 0.0, "negative precipitation value: {v}");
        }
    }
}
