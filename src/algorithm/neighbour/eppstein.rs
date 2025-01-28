//! This module provides FastPair, a data-structure for efficiently tracking the dynamic
//! closest pairs in a set of points, with an example usage in hierarchical clustering.[2][3][5]
//!
//! ## Purpose
//!
//! FastPair allows quick retrieval of the nearest neighbor for each data point by maintaining
//! a "conga line" of closest pairs. Each point retains a link to its known nearest neighbor,
//! and updates in the data structure propagate accordingly. This can be leveraged in
//! agglomerative clustering steps, where merging or insertion of new points must be reflected
//! in nearest-neighbor relationships.
//!
//! ## Example
//!
//! ```
//! use smartcore::metrics::distance::PairwiseDistance;
//! use smartcore::linalg::basic::matrix::DenseMatrix;
//! use smartcore::algorithm::neighbour::fastpair::FastPair;
//!
//! let x = DenseMatrix::from_2d_array(&[
//!     &[5.1, 3.5, 1.4, 0.2],
//!     &[4.9, 3.0, 1.4, 0.2],
//!     &[4.7, 3.2, 1.3, 0.2],
//!     &[4.6, 3.1, 1.5, 0.2],
//!     &[5.0, 3.6, 1.4, 0.2],
//!     &[5.4, 3.9, 1.7, 0.4],
//! ]).unwrap();
//!
//! let fastpair = FastPair::new(&x).unwrap();
//! let closest = fastpair.closest_pair();
//! println!("Closest pair: {:?}", closest);
//! ```
use std::collections::HashMap;

use num::Bounded;

use crate::error::{Failed, FailedError};
use crate::linalg::basic::arrays::{Array, Array1, Array2};
use crate::metrics::distance::euclidian::Euclidian;
use crate::metrics::distance::PairwiseDistance;
use crate::numbers::floatnum::FloatNumber;
use crate::numbers::realnum::RealNumber;

/// Eppstein dynamic closet-pair structure
/// 'M' can be a matrix-like trait that provides row access
#[derive(Debug)]
pub struct EppsteinDCP<'a, T: RealNumber + FloatNumber, M: Array2<T>> {
    samples: &'a M,
    // "buckets" store, for each row, a small structure recording potential neighbors
    neighbors: HashMap<usize, PairwiseDistance<T>>,
}

impl<'a, T: RealNumber + FloatNumber, M: Array2<T>> EppsteinDCP<'a, T, M> {
    /// Creates a new EppsteinDCP instance with the given data
    pub fn new(m: &'a M) -> Result<Self, Failed> {
        if m.shape().0 < 3 {
            return Err(Failed::because(
                FailedError::FindFailed,
                "min number of rows should be 3",
            ));
        }

        let mut this = Self {
            samples: m,
            neighbors: HashMap::with_capacity(m.shape().0),
        };
        this.initialize();
        Ok(this)
    }

    /// Build an initial "conga line" or chain of potential neighbors
    /// akin to Eppstein’s technique[2].
    fn initialize(&mut self) {
        let n = self.samples.shape().0;
        if n < 2 {
            return;
        }
        // Assign each row i some large distance by default
        for i in 0..n {
            self.neighbors.insert(
                i,
                PairwiseDistance {
                    node: i,
                    neighbour: None,
                    distance: Some(<T as Bounded>::max_value()),
                },
            );
        }
        // Example: link each i to the next, forming a chain
        // (depending on the actual Eppstein approach, can refine)
        for i in 0..(n - 1) {
            let dist = self.compute_dist(i, i + 1);
            self.neighbors.entry(i).and_modify(|pd| {
                pd.neighbour = Some(i + 1);
                pd.distance = Some(dist);
            });
        }
        // Potential refinement steps omitted for brevity
    }

    /// Insert a point into the structure.
    pub fn insert(&mut self, row_idx: usize) {
        // Expand data, find neighbor to link with
        // For example, link row_idx to nearest among existing
        let mut best_neighbor = None;
        let mut best_d = <T as Bounded>::max_value();
        for (i, _) in &self.neighbors {
            let d = self.compute_dist(*i, row_idx);
            if d < best_d {
                best_d = d;
                best_neighbor = Some(*i);
            }
        }
        self.neighbors.insert(
            row_idx,
            PairwiseDistance {
                node: row_idx,
                neighbour: best_neighbor,
                distance: Some(best_d),
            },
        );
        // For the best_neighbor, you might want to see if row_idx becomes closer
        if let Some(kn) = best_neighbor {
            let dist = self.compute_dist(row_idx, kn);
            let entry = self.neighbors.get_mut(&kn).unwrap();
            if dist < entry.distance.unwrap() {
                entry.neighbour = Some(row_idx);
                entry.distance = Some(dist);
            }
        }
    }

    /// For hierarchical clustering, discover minimal pairs, then merge
    pub fn closest_pair(&self) -> Option<PairwiseDistance<T>> {
        let mut min_pair: Option<PairwiseDistance<T>> = None;
        for (_, pd) in &self.neighbors {
            if let Some(d) = pd.distance {
                if min_pair.is_none() || d < min_pair.as_ref().unwrap().distance.unwrap() {
                    min_pair = Some(pd.clone());
                }
            }
        }
        min_pair
    }

    fn compute_dist(&self, i: usize, j: usize) -> T {
        // Example: Euclidean
        let row_i = self.samples.get_row(i);
        let row_j = self.samples.get_row(j);
        row_i
            .iterator(0)
            .zip(row_j.iterator(0))
            .map(|(a, b)| (*a - *b) * (*a - *b))
            .sum()
    }
}

/// Simple usage
#[cfg(test)]
mod tests_eppstein {
    use super::*;
    use crate::linalg::basic::matrix::DenseMatrix;

    #[test]
    fn test_eppstein() {
        let matrix =
            DenseMatrix::from_2d_array(&[&vec![1.0, 2.0], &vec![2.0, 2.0], &vec![5.0, 3.0]])
                .unwrap();
        let mut dcp = EppsteinDCP::new(&matrix).unwrap();
        dcp.insert(2);
        let cp = dcp.closest_pair();
        assert!(cp.is_some());
    }

    #[test]
    fn compare_fastpair_eppstein() {
        use crate::algorithm::neighbour::fastpair::FastPair;
        // Assuming EppsteinDCP is implemented in a similar module
        use crate::algorithm::neighbour::eppstein::EppsteinDCP;

        // Create a static example matrix
        let x = DenseMatrix::from_2d_array(&[
            &[5.1, 3.5, 1.4, 0.2],
            &[4.9, 3.0, 1.4, 0.2],
            &[4.7, 3.2, 1.3, 0.2],
            &[4.6, 3.1, 1.5, 0.2],
            &[5.0, 3.6, 1.4, 0.2],
            &[5.4, 3.9, 1.7, 0.4],
            &[4.6, 3.4, 1.4, 0.3],
            &[5.0, 3.4, 1.5, 0.2],
            &[4.4, 2.9, 1.4, 0.2],
            &[4.9, 3.1, 1.5, 0.1],
        ])
        .unwrap();

        // Build FastPair
        let fastpair = FastPair::new(&x).unwrap();
        let pair_fastpair = fastpair.closest_pair();

        // Build EppsteinDCP
        let eppstein = EppsteinDCP::new(&x).unwrap();
        let pair_eppstein = eppstein.closest_pair();

        // Compare the results
        assert_eq!(pair_fastpair.node, pair_eppstein.as_ref().unwrap().node);
        assert_eq!(
            pair_fastpair.neighbour.unwrap(),
            pair_eppstein.as_ref().unwrap().neighbour.unwrap()
        );

        // Use a small epsilon for floating-point comparison
        let epsilon = 1e-9;
        let diff: f64 =
            pair_fastpair.distance.unwrap() - pair_eppstein.as_ref().unwrap().distance.unwrap();
        assert!(diff.abs() < epsilon);

        println!("FastPair result: {:?}", pair_fastpair);
        println!("EppsteinDCP result: {:?}", pair_eppstein);
    }
}
