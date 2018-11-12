extern crate std;

use data::SparseVector;
use order_stat::kth_by;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::f32;

/// Pick initial centroids for K-means clustering, using K-means++.
fn pick_centroids(vectors: &[&SparseVector], k: u32) -> (Vec<SparseVector>, Vec<usize>) {
    assert!(k > 0);
    assert!(!vectors.is_empty());

    let mut centroids: Vec<SparseVector> = Vec::with_capacity(k as usize);
    let mut partitions = vec![0usize; vectors.len()];
    let mut cos_sims = vec![0f32; vectors.len()];

    while centroids.len() < k as usize {
        let c = {
            // Randomly pick the centroid using the weighted probability distribution
            let weights: Vec<_> = cos_sims
                .iter()
                .map(|&s| {
                    // - If it's very close to a centroid already, just don't pick it;
                    // - Otherwise, pick with probability proportional to a distance that respects
                    //   triangle inequality. For details, see:
                    //   Endo Y., Miyamoto S. (2015) Spherical k-Means++ Clustering. In: Modeling
                    //   Decisions for Artificial Intelligence. MDAI 2015. Lecture Notes in Computer
                    //   Science, vol 9321. Springer, Cham. https://doi.org/10.1007/978-3-319-23240-9_9
                    if s > 1. - 1e-4 {
                        0.
                    } else {
                        1.5 - s
                    }
                }).collect();
            if let Ok(distribution) = WeightedIndex::new(weights) {
                vectors[distribution.sample(&mut thread_rng())].clone()
            } else {
                // All weights are zero, break out of the loop
                break;
            }
        };

        // Update cosine similarities with nearest centroid
        for (v, curr_s, curr_p) in izip!(vectors, &mut cos_sims, &mut partitions) {
            let s = v.dot(&c);
            if s > *curr_s {
                assert!(s < 1. + 1e-4);
                *curr_s = s.min(1.);
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
    centroid_preserve_ratio: f32,
    centroid_min_n_preserve: usize,
) -> Vec<SparseVector> {
    debug_assert!(!vectors.is_empty());
    debug_assert!(partitions.len() == vectors.len());

    // Compute centroids for each partition
    let compute_centroid = |indices: &Vec<usize>| {
        if indices.is_empty() {
            // NB: empty partitions are filtered out, therefore partition indices might
            //     not match with centroids
            return None;
        }

        // Sum up vectors with the given indices
        let mut index_value_pairs: Vec<_> = {
            let mut index_to_value = HashMap::new();
            for &i in indices {
                for &(index, value) in &vectors[i].entries {
                    let ref_v = index_to_value.entry(index).or_insert(0.);
                    *ref_v += value;
                }
            }
            index_to_value.into_iter().collect()
        };

        // Prune smaller values if necessary
        let n_preserve =
            ((index_value_pairs.len() as f32 * centroid_preserve_ratio) as f32).ceil() as usize;
        let n_preserve = n_preserve
            .max(centroid_min_n_preserve)
            .min(index_value_pairs.len());
        assert!(index_value_pairs.is_empty() || n_preserve > 0);
        if n_preserve < index_value_pairs.len() {
            kth_by(&mut index_value_pairs, n_preserve - 1, |l, r| {
                let (_, lv) = l;
                let (_, rv) = r;
                rv.partial_cmp(lv).unwrap()
            });
            index_value_pairs.truncate(n_preserve);
        }

        // Create centroid vector from the entries that are left
        let mut sv = SparseVector::from(index_value_pairs);
        sv.l2_normalize();
        Some(sv)
    };
    let centroids: Vec<_> = {
        let mut indices_per_partition: Vec<Vec<usize>> = Vec::new();
        for (i, &p) in partitions.iter().enumerate() {
            if indices_per_partition.len() < p + 1 {
                indices_per_partition.resize(p + 1, Vec::new());
            }
            indices_per_partition[p].push(i);
        }

        indices_per_partition
            .iter()
            .filter_map(compute_centroid)
            .collect()
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
    let centroids = compute_centroids_per_partition(vectors, partitions, 1., 1);
    reassign_partitions(vectors, &centroids, partitions);
    centroids
}

/// Run spherical K-means on a set of data points.
pub fn skmeans(vectors: &[&SparseVector], k: u32, n_iter: u32) -> (Vec<SparseVector>, Vec<usize>) {
    let (mut centroids, mut partitions) = pick_centroids(vectors, k);
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
        let mut vectors = [
            SparseVector::from(vec![(1, 1.), (2, 2.)]),
            SparseVector::from(vec![(11, 12.), (22, 23.)]),
            SparseVector::from(vec![(111, 123.), (222, 234.)]),
        ];
        for v in vectors.iter_mut() {
            v.l2_normalize();
        }

        let vector_refs = [
            &vectors[0].clone(), // 0
            &vectors[1].clone(), // 1
            &vectors[2].clone(), // 2
            &vectors[0].clone(), // 3
            &vectors[1].clone(), // 4
            &vectors[0].clone(), // 5
        ];

        // Run K-means++
        let (centroids, partitions) = super::pick_centroids(&vector_refs, 3);

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
            super::compute_centroids_per_partition(&vectors, &mut partitions, 1., 1)
        );

        // Prune vectors to have max length of 2
        assert_eq!(
            vec![
                SparseVector::from(vec![
                    (1, 1. / (1f32 + 0.64).sqrt()),
                    (5, 0.8 / (1f32 + 0.64).sqrt()),
                ]),
                SparseVector::from(vec![
                    (3, 0.8 / (1f32 + 0.64).sqrt()),
                    (5, 1. / (1f32 + 0.64).sqrt()),
                ]),
                SparseVector::from(vec![(3, 1.)]),
            ],
            super::compute_centroids_per_partition(&vectors, &mut partitions, 0.1, 2),
        );

        // Ditto
        assert_eq!(
            vec![
                SparseVector::from(vec![
                    (1, 1. / (1f32 + 0.64).sqrt()),
                    (5, 0.8 / (1f32 + 0.64).sqrt()),
                ]),
                SparseVector::from(vec![
                    (3, 0.8 / (1f32 + 0.64).sqrt()),
                    (5, 1. / (1f32 + 0.64).sqrt()),
                ]),
                SparseVector::from(vec![(3, 1.)]),
            ],
            super::compute_centroids_per_partition(&vectors, &mut partitions, 0.5, 1),
        );
    }

    #[test]
    fn test_reassign_partitions() {
        let mut vectors = [
            SparseVector::from(vec![(1, 1.), (2, 2.)]),
            SparseVector::from(vec![(11, 12.), (22, 23.)]),
            SparseVector::from(vec![(111, 123.), (222, 234.)]),
            SparseVector::from(vec![(1111, 123.), (2222, 2345.)]),
        ];
        for v in vectors.iter_mut() {
            v.l2_normalize();
        }

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
