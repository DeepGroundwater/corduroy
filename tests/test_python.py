"""Offline tests for the corduroy Python bindings.

Run:
    maturin develop
    pytest tests/test_python.py -v
"""

import numpy as np
import pytest


def test_import():
    import corduroy

    assert hasattr(corduroy, "BoundingBox")
    assert hasattr(corduroy, "PrecipitationField")
    assert hasattr(corduroy, "ConservativeRegridder")
    assert hasattr(corduroy, "Era5Dataset")
    assert hasattr(corduroy, "fetch_precipitation")
    assert hasattr(corduroy, "make_precipitation_field")
    assert hasattr(corduroy, "render_heatmap")
    assert hasattr(corduroy, "render_heatmap_to_file")
    assert hasattr(corduroy, "datetime_to_time_index")
    assert hasattr(corduroy, "fetch_era5_global")


# -- BoundingBox -------------------------------------------------------------


class TestBoundingBox:
    def test_valid(self):
        import corduroy

        bbox = corduroy.BoundingBox(35.0, 45.0, -90.0, -75.0)
        assert bbox.lat_min == 35.0
        assert bbox.lat_max == 45.0
        assert bbox.lon_min == -90.0
        assert bbox.lon_max == -75.0

    def test_repr(self):
        import corduroy

        bbox = corduroy.BoundingBox(35.0, 45.0, -90.0, -75.0)
        r = repr(bbox)
        assert "BoundingBox" in r
        assert "35" in r

    def test_invalid_lat_range(self):
        import corduroy

        with pytest.raises(RuntimeError):
            corduroy.BoundingBox(50.0, 30.0, -90.0, -75.0)

    def test_invalid_latitude(self):
        import corduroy

        with pytest.raises(RuntimeError):
            corduroy.BoundingBox(-91.0, 45.0, -90.0, -75.0)

    def test_invalid_longitude(self):
        import corduroy

        with pytest.raises(RuntimeError):
            corduroy.BoundingBox(35.0, 45.0, -181.0, -75.0)


# -- PrecipitationField from numpy -------------------------------------------


class TestMakeField:
    def test_from_numpy(self):
        import corduroy

        data = np.random.rand(10, 15).astype(np.float32) * 0.01
        lats = np.linspace(45.0, 42.75, 10)
        lons = np.linspace(270.0, 273.5, 15)

        field = corduroy.make_precipitation_field(
            data, lats, lons, "2023-06-15T12:00:00"
        )

        assert field.shape == (10, 15)
        assert field.datetime == "2023-06-15T12:00:00"

    def test_data_roundtrip(self):
        import corduroy

        original = np.array([[0.001, 0.002], [0.003, 0.004]], dtype=np.float32)
        lats = np.array([45.0, 44.75])
        lons = np.array([270.0, 270.25])

        field = corduroy.make_precipitation_field(
            original, lats, lons, "2023-01-01T00:00:00"
        )
        np.testing.assert_array_almost_equal(field.data, original)

    def test_coords_roundtrip(self):
        import corduroy

        data = np.zeros((3, 2), dtype=np.float32)
        lats = np.array([45.0, 44.75, 44.5])
        lons = np.array([270.0, 270.25])

        field = corduroy.make_precipitation_field(
            data, lats, lons, "2023-01-01T00:00:00"
        )
        np.testing.assert_array_almost_equal(field.latitudes, lats)
        np.testing.assert_array_almost_equal(field.longitudes, lons)

    def test_shape_mismatch_lat(self):
        import corduroy

        data = np.zeros((3, 2), dtype=np.float32)
        lats = np.array([45.0, 44.75])  # wrong: 2 != 3
        lons = np.array([270.0, 270.25])

        with pytest.raises(ValueError, match="latitudes length"):
            corduroy.make_precipitation_field(data, lats, lons, "2023-01-01T00:00:00")

    def test_shape_mismatch_lon(self):
        import corduroy

        data = np.zeros((3, 2), dtype=np.float32)
        lats = np.array([45.0, 44.75, 44.5])
        lons = np.array([270.0])  # wrong: 1 != 2

        with pytest.raises(ValueError, match="longitudes length"):
            corduroy.make_precipitation_field(data, lats, lons, "2023-01-01T00:00:00")

    def test_invalid_datetime(self):
        import corduroy

        data = np.zeros((2, 2), dtype=np.float32)
        lats = np.array([45.0, 44.75])
        lons = np.array([270.0, 270.25])

        with pytest.raises(ValueError, match="invalid datetime"):
            corduroy.make_precipitation_field(data, lats, lons, "not-a-date")


# -- value_range --------------------------------------------------------------


class TestValueRange:
    def test_basic(self):
        import corduroy

        data = np.array([[0.001, 0.005], [0.0, 0.010]], dtype=np.float32)
        lats = np.array([45.0, 44.75])
        lons = np.array([270.0, 270.25])

        field = corduroy.make_precipitation_field(data, lats, lons, "2023-01-01T00:00:00")
        vmin, vmax = field.value_range()
        assert abs(vmin - 0.0) < 1e-6
        assert abs(vmax - 0.010) < 1e-6

    def test_with_nan(self):
        import corduroy

        data = np.array([[np.nan, 0.005], [0.001, np.nan]], dtype=np.float32)
        lats = np.array([45.0, 44.75])
        lons = np.array([270.0, 270.25])

        field = corduroy.make_precipitation_field(data, lats, lons, "2023-01-01T00:00:00")
        vmin, vmax = field.value_range()
        assert abs(vmin - 0.001) < 1e-6
        assert abs(vmax - 0.005) < 1e-6


# -- Render pipeline ----------------------------------------------------------


class TestRender:
    def test_render_to_bytes(self):
        import corduroy

        data = np.random.rand(10, 10).astype(np.float32) * 0.01
        lats = np.linspace(45.0, 42.75, 10)
        lons = np.linspace(270.0, 272.25, 10)

        field = corduroy.make_precipitation_field(data, lats, lons, "2023-06-15T12:00:00")
        png_bytes = corduroy.render_heatmap(field)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 100
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_to_file(self, tmp_path):
        import corduroy

        data = np.random.rand(5, 8).astype(np.float32) * 0.01
        lats = np.linspace(45.0, 44.0, 5)
        lons = np.linspace(270.0, 271.75, 8)

        field = corduroy.make_precipitation_field(data, lats, lons, "2023-06-15T12:00:00")
        out = tmp_path / "test_output.png"
        corduroy.render_heatmap_to_file(field, str(out))

        assert out.exists()
        assert out.stat().st_size > 100

    def test_render_with_nan(self):
        import corduroy

        data = np.full((5, 5), np.nan, dtype=np.float32)
        data[2, 2] = 0.005
        lats = np.linspace(45.0, 44.0, 5)
        lons = np.linspace(270.0, 271.0, 5)

        field = corduroy.make_precipitation_field(data, lats, lons, "2023-01-01T00:00:00")
        png_bytes = corduroy.render_heatmap(field)
        assert png_bytes[:4] == b"\x89PNG"

    def test_render_zeros(self):
        import corduroy

        data = np.zeros((8, 8), dtype=np.float32)
        lats = np.linspace(45.0, 43.25, 8)
        lons = np.linspace(270.0, 271.75, 8)

        field = corduroy.make_precipitation_field(data, lats, lons, "2023-01-01T00:00:00")
        png_bytes = corduroy.render_heatmap(field)
        assert png_bytes[:4] == b"\x89PNG"


# -- datetime_to_time_index ---------------------------------------------------


class TestTimeIndex:
    def test_epoch(self):
        import corduroy

        assert corduroy.datetime_to_time_index("1940-01-01T00:00:00") == 0

    def test_one_day(self):
        import corduroy

        assert corduroy.datetime_to_time_index("1940-01-02T00:00:00") == 24

    def test_known_date(self):
        import corduroy

        # 2023-06-15T12:00:00 should be some large number of hours
        idx = corduroy.datetime_to_time_index("2023-06-15T12:00:00")
        assert idx > 700_000

    def test_before_epoch(self):
        import corduroy

        with pytest.raises(RuntimeError):
            corduroy.datetime_to_time_index("1939-12-31T23:00:00")

    def test_invalid_format(self):
        import corduroy

        with pytest.raises(ValueError, match="invalid datetime"):
            corduroy.datetime_to_time_index("June 15, 2023")


# -- ConservativeRegridder ---------------------------------------------------


class TestRegridder:
    def test_identity(self):
        import corduroy

        lats = np.linspace(45.0, 36.0, 10)
        lons = np.linspace(0.0, 9.0, 10)

        rg = corduroy.ConservativeRegridder(lats, lons, lats, lons)
        assert rg.target_shape == (10, 10)

        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = rg.regrid(data)
        np.testing.assert_allclose(result, data, atol=1e-3)

    def test_coarsening_uniform(self):
        import corduroy

        src_lats = np.linspace(45.0, 35.5, 20)
        src_lons = np.linspace(0.0, 9.5, 20)
        tgt_lats = np.linspace(44.75, 35.75, 10)
        tgt_lons = np.linspace(0.25, 9.25, 10)

        rg = corduroy.ConservativeRegridder(src_lats, src_lons, tgt_lats, tgt_lons)
        assert rg.target_shape == (10, 10)

        # Uniform field should be preserved
        data = np.full((20, 20), 5.0, dtype=np.float32)
        result = rg.regrid(data)
        np.testing.assert_allclose(result, 5.0, atol=0.1)

    def test_nan_renormalization(self):
        import corduroy

        lats = np.linspace(45.0, 42.0, 4)
        lons = np.linspace(0.0, 3.0, 4)
        tgt_lats = np.linspace(44.5, 42.5, 2)
        tgt_lons = np.linspace(0.5, 2.5, 2)

        rg = corduroy.ConservativeRegridder(lats, lons, tgt_lats, tgt_lons)
        data = np.ones((4, 4), dtype=np.float32)
        data[0, 0] = np.nan

        result = rg.regrid(data)
        assert not np.any(np.isnan(result)), "valid source cells should produce non-NaN output"
        np.testing.assert_allclose(result, 1.0, atol=0.1)

    def test_all_nan(self):
        import corduroy

        lats = np.linspace(45.0, 42.0, 4)
        lons = np.linspace(0.0, 3.0, 4)

        rg = corduroy.ConservativeRegridder(lats, lons, lats, lons)
        data = np.full((4, 4), np.nan, dtype=np.float32)

        result = rg.regrid(data)
        assert np.all(np.isnan(result))

    def test_shape_mismatch(self):
        import corduroy

        lats = np.linspace(45.0, 42.0, 4)
        lons = np.linspace(0.0, 3.0, 4)

        rg = corduroy.ConservativeRegridder(lats, lons, lats, lons)
        wrong = np.zeros((3, 4), dtype=np.float32)

        with pytest.raises(RuntimeError, match="shape"):
            rg.regrid(wrong)

    def test_non_monotonic_rejected(self):
        import corduroy

        bad_lats = np.array([45.0, 44.0, 46.0, 43.0])
        lons = np.linspace(0.0, 3.0, 4)

        with pytest.raises(RuntimeError):
            corduroy.ConservativeRegridder(bad_lats, lons, bad_lats, lons)

    def test_repr(self):
        import corduroy

        lats = np.linspace(45.0, 42.0, 4)
        lons = np.linspace(0.0, 3.0, 4)

        rg = corduroy.ConservativeRegridder(lats, lons, lats, lons)
        assert "ConservativeRegridder" in repr(rg)
        assert "4" in repr(rg)


# -- fetch_era5_global (offline) ---------------------------------------------


class TestFetchEra5GlobalOffline:
    def test_invalid_time_start(self):
        import corduroy

        with pytest.raises(ValueError, match="invalid time_start"):
            corduroy.fetch_era5_global(
                ["temperature"], "bad-date", "2023-06-15T12:00:00"
            )

    def test_invalid_time_end(self):
        import corduroy

        with pytest.raises(ValueError, match="invalid time_end"):
            corduroy.fetch_era5_global(
                ["temperature"], "2023-06-15T00:00:00", "bad-date"
            )

    def test_default_time_step(self):
        """Verify the default time_step_hours=6 is accepted."""
        import corduroy
        import inspect

        sig = inspect.signature(corduroy.fetch_era5_global)
        assert sig.parameters["time_step_hours"].default == 6


# -- fetch_era5_global (network) ---------------------------------------------


@pytest.mark.skipif(
    not pytest.importorskip("gcsfs", reason="gcsfs required"),
    reason="network test",
)
class TestFetchEra5GlobalNetwork:
    """Integration tests that actually fetch from GCS. Run with: pytest -k Network"""

    @pytest.mark.slow
    def test_fetch_single_surface_var(self):
        """Fetch one surface variable for one timestep."""
        import corduroy

        ds = corduroy.fetch_era5_global(
            variables=["sea_ice_cover"],
            time_start="2023-06-15T00:00:00",
            time_end="2023-06-15T00:00:00",
            time_step_hours=1,
        )

        assert "sea_ice_cover" in ds.variable_names
        arr = ds.get("sea_ice_cover")
        assert arr.ndim == 3  # (time, lat, lon)
        assert arr.shape[0] == 1  # 1 timestep
        assert arr.shape[1] == 721
        assert arr.shape[2] == 1440
        assert len(ds.times) == 1
        assert len(ds.latitudes) == 721
        assert len(ds.longitudes) == 1440
        assert len(ds.levels) == 37
        assert ds.dims("sea_ice_cover") == ["time", "latitude", "longitude"]
        assert "Era5Dataset" in repr(ds)


# -- regrid_dataset (offline) ------------------------------------------------


class TestRegridDataset:
    def _make_mock_dataset(self):
        """Build a minimal Era5Dataset-like object using the regridder's identity case."""
        import corduroy

        # 4x4 grid, 1 timestep
        lats = np.linspace(45.0, 42.0, 4)
        lons = np.linspace(0.0, 3.0, 4)
        # Need a real Era5Dataset — build one by fetching? No, construct manually.
        # We can't construct Era5Dataset from Python, so we test via the regridder
        # on synthetic arrays. Use fetch_era5_global in network tests instead.
        return lats, lons

    def test_regrid_dataset_identity(self):
        """regrid_dataset on a fetched dataset with identity grid preserves values."""
        # This test requires network. Mark it appropriately.
        pytest.importorskip("gcsfs", reason="gcsfs required")
        import corduroy

        ds = corduroy.fetch_era5_global(
            variables=["sea_ice_cover"],
            time_start="2023-06-15T00:00:00",
            time_end="2023-06-15T00:00:00",
            time_step_hours=1,
        )

        # Identity regrid: same source and target grid
        rg = corduroy.ConservativeRegridder(
            ds.latitudes, ds.longitudes, ds.latitudes, ds.longitudes
        )
        result = rg.regrid_dataset(ds)

        assert result.variable_names == ds.variable_names
        assert len(result.latitudes) == len(ds.latitudes)
        assert len(result.longitudes) == len(ds.longitudes)
        src = ds.get("sea_ice_cover")
        dst = result.get("sea_ice_cover")
        assert src.shape == dst.shape
        # Values should be close (identity regrid)
        valid = ~np.isnan(src) & ~np.isnan(dst)
        np.testing.assert_allclose(dst[valid], src[valid], atol=1e-2)

    def test_regrid_dataset_coarsening(self):
        """regrid_dataset coarsens a fetched dataset to a smaller grid."""
        pytest.importorskip("gcsfs", reason="gcsfs required")
        import corduroy

        ds = corduroy.fetch_era5_global(
            variables=["sea_ice_cover"],
            time_start="2023-06-15T00:00:00",
            time_end="2023-06-15T00:00:00",
            time_step_hours=1,
        )

        # Coarsen to ~1 degree (181 lat x 360 lon)
        tgt_lats = np.linspace(90.0, -90.0, 181)
        tgt_lons = np.linspace(0.0, 359.0, 360)

        rg = corduroy.ConservativeRegridder(
            ds.latitudes, ds.longitudes, tgt_lats, tgt_lons
        )
        result = rg.regrid_dataset(ds)

        arr = result.get("sea_ice_cover")
        assert arr.shape == (1, 181, 360)
        assert len(result.latitudes) == 181
        assert len(result.longitudes) == 360
