//! Msgpack wire format for the WebSocket snapshot, "column-packed".
//!
//! A JSON snapshot weighs ~0.5 KB/cell, more than half of it in *field
//! names* repeated for each of the N cells. Encoding the same snapshot in
//! msgpack with named maps would therefore gain almost nothing. Here the
//! cell field names are transmitted **once per frame** in `cell_fields`,
//! and each cell becomes a positional array in the same order, ~3x more
//! compact than JSON, while staying self-descriptive: the consumer zips
//! `cell_fields` with each row without a hardcoded schema.
//!
//! The header (`tick`, totals, `species_order`…) stays a named map: its
//! cost is paid once per frame, not per cell.
//!
//! The `GridState` from `hexsim-core` remains the single source of truth
//! for the content; this module is only its transport projection. The
//! `cell_row_matches_named_serialization` test breaks if a field is added
//! to `CellSnapshot` without being added here (and vice versa).

use bytes::Bytes;
use hexsim_core::snapshot::{CellSnapshot, GridState};
use serde::ser::{Serialize, SerializeSeq, SerializeStruct, SerializeTuple, Serializer};

/// Cell field names, in the exact order of `CellRow`'s elements.
pub const CELL_FIELDS: [&str; 31] = [
    "q",
    "r",
    "elevation",
    "temperature",
    "water_level",
    "water_capacity",
    "humidity_surface",
    "humidity_upper",
    "cloud_water",
    "groundwater",
    "snow_level",
    "permeability",
    "vegetation",
    "dominant_species",
    "species_mix",
    "stand_age",
    "fire_intensity",
    "is_open_water",
    "is_raining",
    "rain_amount",
    "snow_amount",
    "outflow_flux",
    "flow_vec_x",
    "flow_vec_y",
    "edge_flux",
    "wind_x",
    "wind_y",
    "synoptic_h",
    "synoptic_u",
    "synoptic_v",
    "illumination",
];

/// A cell serialized as a positional array, order = `CELL_FIELDS`.
struct CellRow<'a>(&'a CellSnapshot);

impl Serialize for CellRow<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let c = self.0;
        let mut row = serializer.serialize_tuple(CELL_FIELDS.len())?;
        row.serialize_element(&c.q)?;
        row.serialize_element(&c.r)?;
        row.serialize_element(&c.elevation)?;
        row.serialize_element(&c.temperature)?;
        row.serialize_element(&c.water_level)?;
        row.serialize_element(&c.water_capacity)?;
        row.serialize_element(&c.humidity_surface)?;
        row.serialize_element(&c.humidity_upper)?;
        row.serialize_element(&c.cloud_water)?;
        row.serialize_element(&c.groundwater)?;
        row.serialize_element(&c.snow_level)?;
        row.serialize_element(&c.permeability)?;
        row.serialize_element(&c.vegetation)?;
        row.serialize_element(&c.dominant_species)?;
        row.serialize_element(&c.species_mix)?;
        row.serialize_element(&c.stand_age)?;
        row.serialize_element(&c.fire_intensity)?;
        row.serialize_element(&c.is_open_water)?;
        row.serialize_element(&c.is_raining)?;
        row.serialize_element(&c.rain_amount)?;
        row.serialize_element(&c.snow_amount)?;
        row.serialize_element(&c.outflow_flux)?;
        row.serialize_element(&c.flow_vec_x)?;
        row.serialize_element(&c.flow_vec_y)?;
        row.serialize_element(&c.edge_flux)?;
        row.serialize_element(&c.wind_x)?;
        row.serialize_element(&c.wind_y)?;
        row.serialize_element(&c.synoptic_h)?;
        row.serialize_element(&c.synoptic_u)?;
        row.serialize_element(&c.synoptic_v)?;
        row.serialize_element(&c.illumination)?;
        row.end()
    }
}

/// Cells as a sequence of positional rows.
struct CellRows<'a>(&'a [CellSnapshot]);

impl Serialize for CellRows<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for cell in self.0 {
            seq.serialize_element(&CellRow(cell))?;
        }
        seq.end()
    }
}

/// Transport projection of a `GridState`: named header + `cell_fields`
/// + positional rows. Encode with `rmp_serde::to_vec_named`.
struct WireGridState<'a>(&'a GridState);

impl Serialize for WireGridState<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let g = self.0;
        let mut st = serializer.serialize_struct("GridState", 13)?;
        st.serialize_field("tick", &g.tick)?;
        st.serialize_field("hour_tick", &g.hour_tick)?;
        st.serialize_field("cell_count", &g.cell_count)?;
        st.serialize_field("total_surface_water", &g.total_surface_water)?;
        st.serialize_field("total_humidity", &g.total_humidity)?;
        st.serialize_field("total_cloud_water", &g.total_cloud_water)?;
        st.serialize_field("total_precip_this_tick", &g.total_precip_this_tick)?;
        st.serialize_field("total_groundwater", &g.total_groundwater)?;
        st.serialize_field("total_snow", &g.total_snow)?;
        st.serialize_field("species_order", &g.species_order)?;
        st.serialize_field("edge_flux_max", &g.edge_flux_max)?;
        // Slice, not array: serde only implements Serialize for [T; 0..=32].
        st.serialize_field("cell_fields", CELL_FIELDS.as_slice())?;
        st.serialize_field("cells", &CellRows(&g.cells))?;
        st.end()
    }
}

/// Encodes a snapshot for the WS. `Bytes`: broadcast frames are cloned
/// per subscribed client, we want an O(1) clone even at 10 MB.
///
/// # Errors
///
/// Returns the underlying `MessagePack` encode error if serialization
/// fails. This doesn't happen on a valid `GridState` (every field is a
/// plain number, string or vector, none of which refuse encoding), but
/// the API stays honest rather than defaulting to an empty `Bytes` on
/// failure: that used to hand the caller a zero-byte frame, which the
/// server would then broadcast in silence — the front-end's msgpack
/// decoder threw on an unrelated later frame, with nothing in the logs
/// pointing back to the encode that actually failed.
pub fn encode_snapshot(state: &GridState) -> Result<Bytes, rmp_serde::encode::Error> {
    rmp_serde::to_vec_named(&WireGridState(state)).map(Bytes::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// A `CellSnapshot` where every field has a distinct value: swapping two
    /// fields in `CellRow` would make the equality check fail.
    fn sample_cell() -> CellSnapshot {
        serde_json::from_value(serde_json::json!({
            "q": 3,
            "r": -7,
            "elevation": 101.0,
            "temperature": 102.0,
            "water_level": 103.0,
            "water_capacity": 104.0,
            "humidity_surface": 106.0,
            "humidity_upper": 107.0,
            "cloud_water": 108.0,
            "groundwater": 109.0,
            "snow_level": 110.0,
            "permeability": 111.0,
            "vegetation": 112.0,
            "dominant_species": "oak_pubescent",
            "species_mix": [1.0, 2.0, 3.0, 4.0, 5.0],
            "stand_age": 113.0,
            "fire_intensity": 114.0,
            "is_open_water": true,
            "is_raining": false,
            "rain_amount": 115.0,
            "snow_amount": 116.0,
            "outflow_flux": 117.0,
            "flow_vec_x": 118.0,
            "flow_vec_y": 119.0,
            "edge_flux": [11, 22, 33, 44, 55, 66],
            "wind_x": 120.0,
            "wind_y": 121.0,
            "synoptic_h": 122.0,
            "synoptic_u": 123.0,
            "synoptic_v": 124.0,
            "illumination": 127.0,
        }))
        .expect("CellSnapshot from JSON: a field added to the struct must be added here")
    }

    /// Anti-desync guard: the positional row contains exactly the fields of
    /// the derived serialization of `CellSnapshot`, in the order announced
    /// by `CELL_FIELDS`. Adding a field to the struct without updating
    /// `CellRow` + `CELL_FIELDS` breaks this test.
    #[test]
    fn cell_row_matches_named_serialization() {
        let cell = sample_cell();

        let named = rmp_serde::to_vec_named(&cell).expect("named encode");
        let named: Value = rmp_serde::from_slice(&named).expect("named decode");
        let named = named.as_object().expect("named map");

        let row = rmp_serde::to_vec(&CellRow(&cell)).expect("encode ligne");
        let row: Vec<Value> = rmp_serde::from_slice(&row).expect("decode ligne");

        assert_eq!(
            named.len(),
            CELL_FIELDS.len(),
            "CellSnapshot and CELL_FIELDS don't have the same number of fields"
        );
        assert_eq!(row.len(), CELL_FIELDS.len());
        for (i, name) in CELL_FIELDS.iter().enumerate() {
            assert_eq!(
                named.get(*name),
                Some(&row[i]),
                "field '{name}': position {i} of CellRow does not match"
            );
        }
    }

    /// Same guard for the header: `WireGridState` = all fields of
    /// `GridState` (including `cells`, reformatted) + `cell_fields`.
    #[test]
    fn wire_header_matches_named_serialization() {
        let state = GridState {
            tick: 42,
            hour_tick: 1009,
            cell_count: 1,
            total_surface_water: 201.0,
            total_humidity: 202.0,
            total_cloud_water: 203.0,
            total_precip_this_tick: 204.0,
            total_groundwater: 205.0,
            total_snow: 206.0,
            species_order: sample_grid_state_probe().species_order,
            edge_flux_max: 207.0,
            cells: vec![sample_cell()],
        };

        let named = rmp_serde::to_vec_named(&state).expect("named encode");
        let named: Value = rmp_serde::from_slice(&named).expect("named decode");
        let named = named.as_object().expect("named map");

        let wire = rmp_serde::to_vec_named(&WireGridState(&state)).expect("encode wire");
        let wire: Value = rmp_serde::from_slice(&wire).expect("decode wire");
        let wire = wire.as_object().expect("map wire");

        assert_eq!(
            wire.len(),
            named.len() + 1,
            "WireGridState must expose the fields of GridState + cell_fields"
        );
        for (key, value) in named {
            if key == "cells" {
                continue; // reformatted into positional rows, tested above
            }
            assert_eq!(wire.get(key), Some(value), "header: field '{key}'");
        }
        let fields: Vec<Value> = CELL_FIELDS.iter().map(|f| Value::from(*f)).collect();
        assert_eq!(wire.get("cell_fields"), Some(&Value::from(fields)));
    }

    /// Minimal `GridState` from the real engine, for an authentic
    /// `species_order` without copying the constant from `hexsim-core`.
    /// Empty `HydroMaps` for the probes: the snapshot tolerates maps shorter
    /// than the grid (`get(i)` → defaults).
    fn empty_hydro_maps() -> hexsim_core::hydro::HydroMaps<'static> {
        hexsim_core::hydro::HydroMaps {
            discharge: &[],
            flow_vec: &[],
            edge_flux: &[],
        }
    }

    fn sample_grid_state_probe() -> GridState {
        let grid = hexsim_core::grid::HexGrid::from_radius(0);
        grid.snapshot(0, 0, &empty_hydro_maps(), &Vec::new(), &Vec::new())
    }

    /// The wire format must clearly beat JSON on a real world, that's its
    /// only reason to exist. Threshold at 45%: measured 27% on fresh
    /// terrain R=15 and 23% on a 30-day lived-in world (139 B/cell
    /// constant in wire, f32 msgpack weighs 5 B regardless of the value;
    /// JSON bloats to 584 B/cell with populated floats). The margin
    /// absorbs content drift without letting a regression toward named
    /// msgpack (~95%) slip through.
    #[test]
    fn wire_is_much_smaller_than_json() {
        let mut grid = hexsim_core::grid::HexGrid::from_radius(15);
        hexsim_core::terrain::generate_terrain(
            &mut grid,
            &hexsim_core::terrain::TerrainParams::default(),
        );
        let state = grid.snapshot(0, 0, &empty_hydro_maps(), &Vec::new(), &Vec::new());

        let json_len = serde_json::to_string(&state).expect("json").len();
        let wire_len = encode_snapshot(&state).expect("encode").len();

        println!(
            "wire {wire_len} o vs json {json_len} o: {}%, {} o/cell",
            wire_len * 100 / json_len,
            wire_len / state.cell_count
        );
        assert!(
            wire_len * 100 < json_len * 45,
            "wire {wire_len} o vs json {json_len} o : ratio {}%",
            wire_len * 100 / json_len
        );
    }
}
