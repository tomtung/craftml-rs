extern crate std;

use super::data::SparseVector;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use std::f32;

/// Pick initial centroids for K-means clustering, using K-means++.
fn pick_centroids<R: Rng>(
    vectors: &[&SparseVector],
    k: u32,
    rng: &mut R,
) -> (Vec<SparseVector>, Vec<usize>) {
    assert!(k > 0);
    assert!(!vectors.is_empty());

    let mut centroids: Vec<SparseVector> = Vec::with_capacity(k as usize);
    let mut partitions = vec![0usize; vectors.len()];
    let mut distances = vec![1f32; vectors.len()];

    while centroids.len() < k as usize {
        let c = {
            // Randomly pick the centroid using the weighted probability distribution
            let weights: Vec<_> = distances.iter().map(|d| d.powi(2)).collect();
            if let Ok(distribution) = WeightedIndex::new(weights) {
                vectors[distribution.sample(rng)].clone()
            } else {
                // All weights are zero, break out of the loop
                break;
            }
        };

        // Update distance to nearest centroid
        for (v, curr_d, curr_p) in izip!(vectors, &mut distances, &mut partitions) {
            let d = 1. - v.dot(&c);
            if d < *curr_d {
                assert!(d > -1e-4);
                *curr_d = d.max(0.);
                *curr_p = centroids.len();
            }
        }

        centroids.push(c);
    }
    assert!(!centroids.is_empty());

    (centroids, partitions)
}

pub fn compute_centroids_per_partition(
    vectors: &[&SparseVector],
    partitions: &[usize],
) -> Vec<SparseVector> {
    debug_assert!(!vectors.is_empty());
    debug_assert!(partitions.len() == vectors.len());

    // Compute centroids for each partition
    let centroids: Vec<_> = {
        let mut indices_per_partition: Vec<Vec<usize>> = Vec::new();
        for (i, &p) in partitions.iter().enumerate() {
            if indices_per_partition.len() < p + 1 {
                indices_per_partition.resize(p + 1, Vec::new());
            }
            indices_per_partition[p].push(i);
        }

        indices_per_partition
            .into_iter()
            .filter_map(|indices| {
                if indices.is_empty() {
                    // NB: empty partitions are filtered out, therefore partition indices might
                    //     not match with centroids
                    None
                } else {
                    let vectors = indices.into_iter().map(|i| vectors[i]);
                    Some(SparseVector::sum(vectors).into_l2_normalized())
                }
            }).collect()
    };
    assert!(!centroids.is_empty());

    centroids
}

pub fn reassign_partitions(
    vectors: &[&SparseVector],
    centroids: &[SparseVector],
    partitions: &mut Vec<usize>,
) {
    assert_eq!(vectors.len(), partitions.len());
    vectors
        .par_iter()
        .map(|v| assign_partition(v, centroids.iter()))
        .collect_into_vec(partitions);
}

pub fn assign_partition<'a, T>(vector: &SparseVector, centroid_iter: T) -> usize
where
    T: Iterator<Item = &'a SparseVector>,
{
    let mut curr_d = f32::INFINITY;
    let mut curr_p = 0;

    for (i, c) in centroid_iter.enumerate() {
        let d = 1. - vector.dot(c);
        if d < curr_d {
            assert!(d > -1e-4);
            curr_d = d.max(0.);
            curr_p = i;
        }
    }

    curr_p
}

/// Run 1 iteration of spherical K-means
fn skmeans_iterate(vectors: &[&SparseVector], partitions: &mut Vec<usize>) -> Vec<SparseVector> {
    let centroids = compute_centroids_per_partition(vectors, partitions);
    reassign_partitions(vectors, &centroids, partitions);
    centroids
}

/// Run spherical K-means on a set of data points.
pub fn skmeans<R: Rng>(
    vectors: &[&SparseVector],
    k: u32,
    n_iter: u32,
    rng: &mut R,
) -> (Vec<SparseVector>, Vec<usize>) {
    let (mut centroids, mut partitions) = pick_centroids(vectors, k, rng);
    for _ in 0..n_iter {
        centroids = skmeans_iterate(vectors, &mut partitions);
    }
    (centroids, partitions)
}

#[cfg(test)]
mod tests {
    use super::SparseVector;
    use rand::prelude::*;

    #[test]
    fn test_pick_centroids() {
        // Set up trivial dummy data in which some data points have the same features
        let vectors = [
            SparseVector::from(vec![(1, 1.), (2, 2.)]).into_l2_normalized(),
            SparseVector::from(vec![(11, 12.), (22, 23.)]).into_l2_normalized(),
            SparseVector::from(vec![(111, 123.), (222, 234.)]).into_l2_normalized(),
        ];
        let vector_refs = [
            &vectors[0].clone(), // 0
            &vectors[1].clone(), // 1
            &vectors[2].clone(), // 2
            &vectors[0].clone(), // 3
            &vectors[1].clone(), // 4
            &vectors[0].clone(), // 5
        ];

        // Run K-means++
        let (centroids, partitions) = super::pick_centroids(&vector_refs, 3, &mut thread_rng());

        // We should have 3 centroids
        assert_eq!(3, centroids.len());

        // The same vectors should be assigned to the same centroids equal to themselves
        assert_ne!(partitions[0], partitions[1]);
        assert_ne!(partitions[1], partitions[2]);
        assert_ne!(partitions[0], partitions[2]);
        assert_eq!(partitions[0], partitions[3]);
        assert_eq!(partitions[3], partitions[5]);
        assert_eq!(partitions[1], partitions[4]);

        assert_eq!(centroids[partitions[0]], vectors[0]);
        assert_eq!(centroids[partitions[1]], vectors[1]);
        assert_eq!(centroids[partitions[2]], vectors[2]);
    }

    #[test]
    fn test_skmeans_iterate() {
        let vectors = [
            &SparseVector::from(vec![(1, 1.)]),
            &SparseVector::from(vec![(1, 0.6), (3, 0.8)]),
            &SparseVector::from(vec![(3, 1.)]),
            &SparseVector::from(vec![(3, 0.6), (5, 0.8)]),
            &SparseVector::from(vec![(5, 1.)]),
        ];
        let mut partitions = vec![1, 2, 4, 1, 2];

        // Partitions #0 & #3 are empty, so are ignored
        assert_eq!(
            vec![
                SparseVector::from(vec![
                    (1, 1. / 2f32.sqrt()),
                    (3, 0.6 / 2f32.sqrt()),
                    (5, 0.8 / 2f32.sqrt()),
                ]),
                SparseVector::from(vec![
                    (1, 0.6 / 2f32.sqrt()),
                    (3, 0.8 / 2f32.sqrt()),
                    (5, 1. / 2f32.sqrt()),
                ]),
                SparseVector::from(vec![(3, 1.)]),
            ],
            super::skmeans_iterate(&vectors, &mut partitions)
        );
        assert_eq!(vec![0, 2, 2, 1, 1], partitions);
    }

    #[test]
    fn test_compute_centroids_per_partition() {
        let vectors = [
            &SparseVector::from(vec![(1, 1.)]),
            &SparseVector::from(vec![(1, 0.6), (3, 0.8)]),
            &SparseVector::from(vec![(3, 1.)]),
            &SparseVector::from(vec![(3, 0.6), (5, 0.8)]),
            &SparseVector::from(vec![(5, 1.)]),
        ];
        let mut partitions = [1, 2, 4, 1, 2];

        // Partitions #0 & #3 are empty, so are ignored
        assert_eq!(
            vec![
                SparseVector::from(vec![
                    (1, 1. / 2f32.sqrt()),
                    (3, 0.6 / 2f32.sqrt()),
                    (5, 0.8 / 2f32.sqrt()),
                ]),
                SparseVector::from(vec![
                    (1, 0.6 / 2f32.sqrt()),
                    (3, 0.8 / 2f32.sqrt()),
                    (5, 1. / 2f32.sqrt()),
                ]),
                SparseVector::from(vec![(3, 1.)]),
            ],
            super::compute_centroids_per_partition(&vectors, &mut partitions)
        );
    }

    #[test]
    fn test_reassign_partitions() {
        let vectors = [
            SparseVector::from(vec![(1, 1.), (2, 2.)]).into_l2_normalized(),
            SparseVector::from(vec![(11, 12.), (22, 23.)]).into_l2_normalized(),
            SparseVector::from(vec![(111, 123.), (222, 234.)]).into_l2_normalized(),
            SparseVector::from(vec![(1111, 123.), (2222, 2345.)]).into_l2_normalized(),
        ];
        let vector_refs = [
            &vectors[0].clone(), // 0
            &vectors[1].clone(), // 1
            &vectors[2].clone(), // 2
            &vectors[0].clone(), // 3
            &vectors[1].clone(), // 4
            &vectors[0].clone(), // 5
        ];
        let centroids = [
            vectors[0].clone(), // 0
            vectors[1].clone(), // 1
            vectors[2].clone(), // 2
            vectors[3].clone(), // 3; this centroid won't be used
        ];
        let mut partitions = vec![0; 6];
        super::reassign_partitions(&vector_refs, &centroids, &mut partitions);
        assert_eq!(vec![0, 1, 2, 0, 1, 0], partitions);
    }
}
