use std::path::PathBuf;
use std::sync::OnceLock;

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::domain::{self, BoundingBox, PrecipitationField};
use crate::regrid::ConservativeRegridder;
use crate::{data, plot};

// ---------------------------------------------------------------------------
// Tokio runtime singleton
// ---------------------------------------------------------------------------

fn tokio_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Runtime::new().expect("failed to create tokio runtime")
    })
}

fn to_pyerr<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// PyBoundingBox
// ---------------------------------------------------------------------------

#[pyclass(name = "BoundingBox", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyBoundingBox {
    inner: BoundingBox,
}

#[pymethods]
impl PyBoundingBox {
    #[new]
    fn new(lat_min: f64, lat_max: f64, lon_min: f64, lon_max: f64) -> PyResult<Self> {
        let inner = BoundingBox::new(lat_min, lat_max, lon_min, lon_max).map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    #[getter]
    fn lat_min(&self) -> f64 {
        self.inner.lat_min.value()
    }

    #[getter]
    fn lat_max(&self) -> f64 {
        self.inner.lat_max.value()
    }

    #[getter]
    fn lon_min(&self) -> f64 {
        self.inner.lon_min.value()
    }

    #[getter]
    fn lon_max(&self) -> f64 {
        self.inner.lon_max.value()
    }

    fn __repr__(&self) -> String {
        format!(
            "BoundingBox(lat_min={}, lat_max={}, lon_min={}, lon_max={})",
            self.inner.lat_min.value(),
            self.inner.lat_max.value(),
            self.inner.lon_min.value(),
            self.inner.lon_max.value(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyPrecipitationField
// ---------------------------------------------------------------------------

#[pyclass(name = "PrecipitationField", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyPrecipitationField {
    inner: PrecipitationField,
}

#[pymethods]
impl PyPrecipitationField {
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        numpy::ToPyArray::to_pyarray(&self.inner.data, py)
    }

    #[getter]
    fn latitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.latitudes.clone())
    }

    #[getter]
    fn longitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.longitudes.clone())
    }

    #[getter]
    fn datetime(&self) -> String {
        self.inner.datetime.format("%Y-%m-%dT%H:%M:%S").to_string()
    }

    fn value_range(&self) -> (f32, f32) {
        self.inner.value_range()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.data.dim()
    }

    fn __repr__(&self) -> String {
        let (n_lat, n_lon) = self.inner.data.dim();
        format!(
            "PrecipitationField(shape=({}, {}), datetime='{}')",
            n_lat,
            n_lon,
            self.inner.datetime.format("%Y-%m-%dT%H:%M:%S"),
        )
    }
}

// ---------------------------------------------------------------------------
// PyConservativeRegridder
// ---------------------------------------------------------------------------

#[pyclass(name = "ConservativeRegridder", frozen)]
pub struct PyConservativeRegridder {
    inner: ConservativeRegridder,
}

#[pymethods]
impl PyConservativeRegridder {
    #[new]
    fn new(
        src_lats: PyReadonlyArray1<'_, f64>,
        src_lons: PyReadonlyArray1<'_, f64>,
        tgt_lats: PyReadonlyArray1<'_, f64>,
        tgt_lons: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let inner = ConservativeRegridder::new(
            src_lats.as_slice().map_err(to_pyerr)?,
            src_lons.as_slice().map_err(to_pyerr)?,
            tgt_lats.as_slice().map_err(to_pyerr)?,
            tgt_lons.as_slice().map_err(to_pyerr)?,
        )
        .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    #[getter]
    fn target_shape(&self) -> (usize, usize) {
        self.inner.target_shape()
    }

    fn regrid<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let arr = data.as_array().to_owned();
        let result = self.inner.regrid(&arr).map_err(to_pyerr)?;
        Ok(numpy::ToPyArray::to_pyarray(&result, py))
    }

    fn __repr__(&self) -> String {
        let (tgt_lat, tgt_lon) = self.inner.target_shape();
        format!(
            "ConservativeRegridder(target_shape=({}, {}))",
            tgt_lat, tgt_lon
        )
    }
}

// ---------------------------------------------------------------------------
// PyEra5Dataset
// ---------------------------------------------------------------------------

#[pyclass(name = "Era5Dataset", frozen)]
pub struct PyEra5Dataset {
    inner: data::Era5Dataset,
}

#[pymethods]
impl PyEra5Dataset {
    #[getter]
    fn variable_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.inner.variables.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get a variable as a numpy array.
    ///
    /// Pressure-level variables return shape (time, level, lat, lon).
    /// Surface variables return shape (time, lat, lon).
    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        match self.inner.variables.get(name) {
            Some(data::Era5Array::PressureLevel(arr)) => {
                Ok(numpy::ToPyArray::to_pyarray(arr, py).into_any())
            }
            Some(data::Era5Array::Surface(arr)) => {
                Ok(numpy::ToPyArray::to_pyarray(arr, py).into_any())
            }
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "variable '{}' not found",
                name
            ))),
        }
    }

    /// Get dimension names for a variable.
    fn dims(&self, name: &str) -> PyResult<Vec<String>> {
        match self.inner.variables.get(name) {
            Some(data::Era5Array::PressureLevel(_)) => Ok(vec![
                "time".into(),
                "level".into(),
                "latitude".into(),
                "longitude".into(),
            ]),
            Some(data::Era5Array::Surface(_)) => {
                Ok(vec!["time".into(), "latitude".into(), "longitude".into()])
            }
            None => Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "variable '{}' not found",
                name
            ))),
        }
    }

    #[getter]
    fn times(&self) -> Vec<String> {
        self.inner
            .times
            .iter()
            .map(|t| t.format("%Y-%m-%dT%H:%M:%S").to_string())
            .collect()
    }

    #[getter]
    fn latitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.latitudes.clone())
    }

    #[getter]
    fn longitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.longitudes.clone())
    }

    #[getter]
    fn levels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.levels.clone())
    }

    /// Convert to an xarray.Dataset (requires xarray and pandas installed).
    fn to_xarray<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let xr = py.import("xarray")?;
        let pd = py.import("pandas")?;

        // Time coordinate → pandas DatetimeIndex
        let time_strings = self.times();
        let time_list = pyo3::types::PyList::new(py, &time_strings)?;
        let time_coord = pd.call_method1("to_datetime", (time_list,))?;

        // Build data_vars: {name: (dims_list, numpy_array)}
        let data_vars = pyo3::types::PyDict::new(py);
        for (name, array) in &self.inner.variables {
            let (dims, np_array): (Bound<'py, PyAny>, Bound<'py, PyAny>) = match array {
                data::Era5Array::PressureLevel(arr) => {
                    let d = pyo3::types::PyList::new(
                        py,
                        &["time", "level", "latitude", "longitude"],
                    )?;
                    let a = numpy::ToPyArray::to_pyarray(arr, py).into_any();
                    (d.into_any(), a)
                }
                data::Era5Array::Surface(arr) => {
                    let d =
                        pyo3::types::PyList::new(py, &["time", "latitude", "longitude"])?;
                    let a = numpy::ToPyArray::to_pyarray(arr, py).into_any();
                    (d.into_any(), a)
                }
            };
            let tuple = pyo3::types::PyTuple::new(py, [dims, np_array])?;
            data_vars.set_item(name.as_str(), tuple)?;
        }

        // Coordinates
        let coords = pyo3::types::PyDict::new(py);
        coords.set_item("time", time_coord)?;
        coords.set_item("latitude", self.latitudes(py))?;
        coords.set_item("longitude", self.longitudes(py))?;
        coords.set_item("level", self.levels(py))?;

        // xr.Dataset(data_vars, coords=coords)
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("coords", coords)?;
        xr.call_method("Dataset", (data_vars,), Some(&kwargs))
    }

    fn __repr__(&self) -> String {
        let mut var_names: Vec<&String> = self.inner.variables.keys().collect();
        var_names.sort();
        format!(
            "Era5Dataset(variables={:?}, times={}, lats={}, lons={})",
            var_names,
            self.inner.times.len(),
            self.inner.latitudes.len(),
            self.inner.longitudes.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

/// Fetch ERA5 total precipitation for a bounding box at a specific datetime.
#[pyfunction]
fn fetch_precipitation(bbox: &PyBoundingBox, datetime: &str) -> PyResult<PyPrecipitationField> {
    let dt = chrono::NaiveDateTime::parse_from_str(datetime, "%Y-%m-%dT%H:%M:%S")
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid datetime '{}': {}",
                datetime, e
            ))
        })?;

    let bbox_inner = bbox.inner.clone();
    let rt = tokio_runtime();

    let field = rt.block_on(async {
        let store = data::create_store().await.map_err(to_pyerr)?;
        data::fetch_precipitation(store, &bbox_inner, dt)
            .await
            .map_err(to_pyerr)
    })?;

    Ok(PyPrecipitationField { inner: field })
}

/// Construct a PrecipitationField from numpy arrays (no network required).
#[pyfunction]
fn make_precipitation_field(
    data: PyReadonlyArray2<'_, f32>,
    latitudes: PyReadonlyArray1<'_, f64>,
    longitudes: PyReadonlyArray1<'_, f64>,
    datetime: &str,
) -> PyResult<PyPrecipitationField> {
    let dt = chrono::NaiveDateTime::parse_from_str(datetime, "%Y-%m-%dT%H:%M:%S")
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid datetime '{}': {}",
                datetime, e
            ))
        })?;

    let data_array: Array2<f32> = data.as_array().to_owned();
    let lats: Vec<f64> = latitudes.as_array().to_vec();
    let lons: Vec<f64> = longitudes.as_array().to_vec();

    let (n_lat, n_lon) = data_array.dim();
    if lats.len() != n_lat {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "latitudes length {} != data rows {}",
            lats.len(),
            n_lat
        )));
    }
    if lons.len() != n_lon {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "longitudes length {} != data cols {}",
            lons.len(),
            n_lon
        )));
    }

    Ok(PyPrecipitationField {
        inner: PrecipitationField {
            data: data_array,
            latitudes: lats,
            longitudes: lons,
            datetime: dt,
        },
    })
}

/// Render a PrecipitationField as a PNG and return the bytes.
#[pyfunction]
fn render_heatmap<'py>(
    py: Python<'py>,
    field: &PyPrecipitationField,
) -> PyResult<Bound<'py, PyBytes>> {
    let tmp = std::env::temp_dir().join(format!(
        "_corduroy_render_{}.png",
        std::process::id()
    ));
    plot::render_heatmap(&field.inner, &tmp).map_err(to_pyerr)?;
    let bytes = std::fs::read(&tmp).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("failed to read rendered image: {}", e))
    })?;
    let _ = std::fs::remove_file(&tmp);
    Ok(PyBytes::new(py, &bytes))
}

/// Render a PrecipitationField as a PNG and save to a file.
#[pyfunction]
fn render_heatmap_to_file(field: &PyPrecipitationField, path: &str) -> PyResult<()> {
    plot::render_heatmap(&field.inner, PathBuf::from(path).as_path()).map_err(to_pyerr)
}

/// Convert a datetime string to an ERA5 time index (hours since 1940-01-01).
#[pyfunction]
fn datetime_to_time_index(datetime: &str) -> PyResult<u64> {
    let dt = chrono::NaiveDateTime::parse_from_str(datetime, "%Y-%m-%dT%H:%M:%S")
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid datetime '{}': {}",
                datetime, e
            ))
        })?;
    domain::datetime_to_time_index(dt).map_err(to_pyerr)
}

/// Fetch multiple ERA5 variables at full globe resolution for a time range.
///
/// Variables are fetched in parallel. Each variable is auto-detected as
/// pressure-level (4D) or surface (3D) from its Zarr array dimensionality.
///
/// Returns an Era5Dataset with `.get(name)` to access numpy arrays,
/// and `.to_xarray()` to convert to an xarray.Dataset.
#[pyfunction]
#[pyo3(signature = (variables, time_start, time_end, time_step_hours=6))]
fn fetch_era5_global(
    variables: Vec<String>,
    time_start: &str,
    time_end: &str,
    time_step_hours: u64,
) -> PyResult<PyEra5Dataset> {
    let t_start = chrono::NaiveDateTime::parse_from_str(time_start, "%Y-%m-%dT%H:%M:%S")
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid time_start '{}': {}",
                time_start, e
            ))
        })?;
    let t_end =
        chrono::NaiveDateTime::parse_from_str(time_end, "%Y-%m-%dT%H:%M:%S").map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid time_end '{}': {}",
                time_end, e
            ))
        })?;

    let rt = tokio_runtime();
    let inner = rt.block_on(async {
        let store = data::create_store().await.map_err(to_pyerr)?;
        data::fetch_era5_global(store, &variables, t_start, t_end, time_step_hours)
            .await
            .map_err(to_pyerr)
    })?;

    Ok(PyEra5Dataset { inner })
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
#[pyo3(name = "corduroy")]
fn corduroy_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoundingBox>()?;
    m.add_class::<PyPrecipitationField>()?;
    m.add_class::<PyConservativeRegridder>()?;
    m.add_class::<PyEra5Dataset>()?;
    m.add_function(wrap_pyfunction!(fetch_precipitation, m)?)?;
    m.add_function(wrap_pyfunction!(make_precipitation_field, m)?)?;
    m.add_function(wrap_pyfunction!(render_heatmap, m)?)?;
    m.add_function(wrap_pyfunction!(render_heatmap_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(datetime_to_time_index, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_era5_global, m)?)?;
    Ok(())
}
