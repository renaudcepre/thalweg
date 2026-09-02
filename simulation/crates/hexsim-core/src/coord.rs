use std::ops::{Add, Sub};

use serde::{Deserialize, Serialize};

/// Six axial directions of a hexagon, clockwise starting from east.
pub const DIRECTIONS: [HexCoord; 6] = [
    HexCoord { q: 1, r: 0 },
    HexCoord { q: 1, r: -1 },
    HexCoord { q: 0, r: -1 },
    HexCoord { q: -1, r: 0 },
    HexCoord { q: -1, r: 1 },
    HexCoord { q: 0, r: 1 },
];

/// Unit vector in world coordinates for the hex direction at the given index.
/// Indices correspond to `DIRECTIONS[0..6]`.
#[must_use]
pub fn hex_direction_to_world(index: usize) -> (f32, f32) {
    const SQRT_3_2: f32 = 0.866_025_4;
    const DIRS: [(f32, f32); 6] = [
        (1.0, 0.0),        // East
        (0.5, -SQRT_3_2),  // North-East
        (-0.5, -SQRT_3_2), // North-West
        (-1.0, 0.0),       // West
        (-0.5, SQRT_3_2),  // South-West
        (0.5, SQRT_3_2),   // South-East
    ];
    DIRS[index]
}

/// Translation vectors of the toric lattice for a hexagonal domain of
/// radius `radius`: the centers of 3 of the 6 adjacent copies of the
/// plane's tiling by the hexagon (the other 3 are their opposites).
/// Successive 60° rotations of `v1 = (2R+1, -R)`:
/// `v2 = (R, R+1)`, `v3 = (-(R+1), 2R+1)`.
///
/// `|det(v1, v2)| = (2R+1)(R+1) + R² = 3R² + 3R + 1` = number of cells
/// in the domain → the tiling is **exact**: every point in the plane
/// belongs to exactly one translated copy. This is what makes the toric
/// wrap bijective (`HexGrid::wrap_target`) and the terrain noise periodic
/// (`terrain::TorusNoiseMapping`).
#[must_use]
pub const fn torus_lattice_vectors(radius: i32) -> [HexCoord; 3] {
    [
        HexCoord::new(2 * radius + 1, -radius),
        HexCoord::new(radius, radius + 1),
        HexCoord::new(-(radius + 1), 2 * radius + 1),
    ]
}

/// Axial hexagonal coordinate (q, r).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HexCoord {
    pub q: i32,
    pub r: i32,
}

impl HexCoord {
    #[must_use]
    pub const fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Third cube coordinate, derived: s = -(q + r).
    #[must_use]
    pub const fn s(self) -> i32 {
        -(self.q + self.r)
    }

    /// The 6 neighbors of this cell.
    #[must_use]
    pub fn neighbors(self) -> [HexCoord; 6] {
        DIRECTIONS.map(|d| self + d)
    }

    /// Hexagonal distance (cube metric).
    #[must_use]
    pub fn distance(self, other: Self) -> i32 {
        let diff = self - other;
        (diff.q.abs() + diff.r.abs() + diff.s().abs()) / 2
    }
}

impl Add for HexCoord {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            q: self.q + rhs.q,
            r: self.r + rhs.r,
        }
    }
}

impl Sub for HexCoord {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            q: self.q - rhs.q,
            r: self.r - rhs.r,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn neighbors_of_origin() {
        let origin = HexCoord::new(0, 0);
        let expected: Vec<HexCoord> = DIRECTIONS.to_vec();
        assert_eq!(origin.neighbors().to_vec(), expected);
    }

    #[test]
    fn distance_known_value() {
        let a = HexCoord::new(0, 0);
        let b = HexCoord::new(2, 1);
        assert_eq!(a.distance(b), 3);
    }

    #[test]
    fn distance_to_self_is_zero() {
        let c = HexCoord::new(3, -7);
        assert_eq!(c.distance(c), 0);
    }

    #[test]
    fn distance_is_symmetric() {
        let a = HexCoord::new(1, 2);
        let b = HexCoord::new(-3, 5);
        assert_eq!(a.distance(b), b.distance(a));
    }

    #[test]
    fn distance_to_neighbor_is_one() {
        let origin = HexCoord::new(0, 0);
        for n in origin.neighbors() {
            assert_eq!(origin.distance(n), 1);
        }
    }

    /// 60° clockwise rotation in axial: (q, r) → (-r, q+r).
    fn rot60(c: HexCoord) -> HexCoord {
        HexCoord::new(-c.r, c.q + c.r)
    }

    #[test]
    fn torus_lattice_vectors_are_successive_rotations() {
        for radius in 0..=50 {
            let [v1, v2, v3] = torus_lattice_vectors(radius);
            assert_eq!(rot60(v1), v2, "R={radius}");
            assert_eq!(rot60(v2), v3, "R={radius}");
            // rot60(v3) = -v1: the 6 adjacent copies are indeed ±v1, ±v2, ±v3.
            assert_eq!(rot60(v3), HexCoord::new(-v1.q, -v1.r), "R={radius}");
        }
    }

    #[test]
    fn torus_lattice_determinant_equals_cell_count() {
        // Exact tiling ⟺ the lattice cell area (determinant) equals
        // the number of cells in the hexagonal domain: 3R² + 3R + 1.
        for radius in 0..=50i64 {
            let [v1, v2, _] = torus_lattice_vectors(i32::try_from(radius).unwrap());
            let det = i64::from(v1.q) * i64::from(v2.r) - i64::from(v2.q) * i64::from(v1.r);
            assert_eq!(det, 3 * radius * radius + 3 * radius + 1, "R={radius}");
        }
    }

    proptest! {
        #[test]
        fn prop_distance_to_self(q in -1000..1000i32, r in -1000..1000i32) {
            let c = HexCoord::new(q, r);
            prop_assert_eq!(c.distance(c), 0);
        }

        #[test]
        fn prop_distance_non_negative(
            q1 in -1000..1000i32, r1 in -1000..1000i32,
            q2 in -1000..1000i32, r2 in -1000..1000i32,
        ) {
            let a = HexCoord::new(q1, r1);
            let b = HexCoord::new(q2, r2);
            prop_assert!(a.distance(b) >= 0);
        }

        #[test]
        fn prop_distance_symmetric(
            q1 in -1000..1000i32, r1 in -1000..1000i32,
            q2 in -1000..1000i32, r2 in -1000..1000i32,
        ) {
            let a = HexCoord::new(q1, r1);
            let b = HexCoord::new(q2, r2);
            prop_assert_eq!(a.distance(b), b.distance(a));
        }
    }
}
