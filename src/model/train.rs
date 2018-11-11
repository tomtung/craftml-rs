use super::{CraftmlModel, HashingTrickProjector, Tree, TreeNode};
use data::{DataSet, Label, SparseVector};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use util::draw_async_progress_bar;

#[derive(Debug, Clone)]
pub struct CraftmlTrainer {
    pub n_trees: usize,
    pub n_feature_buckets: u16,
    pub n_label_buckets: u16,
    pub leaf_max_size: usize,
    pub k_clusters: u32,
    pub cluster_sample_size: usize,
    pub n_cluster_iters: u32,
    pub centroid_preserve_ratio: f32,
    pub centroid_min_n_preserve: usize,
}

impl Default for CraftmlTrainer {
    fn default() -> Self {
        Self {
            n_trees: 50,
            n_feature_buckets: 10000,
            n_label_buckets: 10000,
            leaf_max_size: 10,
            k_clusters: 10,
            cluster_sample_size: 20000,
            n_cluster_iters: 2,
            centroid_preserve_ratio: 0.1,
            centroid_min_n_preserve: 5,
        }
    }
}

impl CraftmlTrainer {
    pub fn train(&self, dataset: &DataSet) -> CraftmlModel {
        assert!(self.n_trees > 0);
        assert!(self.n_feature_buckets > 0);
        assert!(self.n_label_buckets > 0);
        assert!(self.leaf_max_size > 0);
        assert!(self.k_clusters > 0);
        assert!(self.cluster_sample_size > 0);
        assert!(self.n_cluster_iters > 0);
        assert!(self.centroid_preserve_ratio >= 0. && self.centroid_preserve_ratio <= 1.);
        assert!(self.centroid_min_n_preserve > 0);

        info!("Training CRAFTML model with parameters {:?}", self);
        let (sender, handle) = draw_async_progress_bar(self.n_trees as u64);
        let start_t = time::precise_time_s();

        let mut trees = Vec::new();
        (0..self.n_trees)
            .into_par_iter()
            .map_with(sender, |sender, _| {
                let tree = self.create_tree_trainer().train(dataset);
                sender.send(1).unwrap();
                tree
            }).collect_into_vec(&mut trees);

        handle.join().unwrap();
        info!(
            "CRAFTML model training complete; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        CraftmlModel { trees }
    }

    fn create_tree_trainer(&self) -> TreeTrainer {
        let mut rng = thread_rng();
        let feature_projector = HashingTrickProjector {
            n_buckets: self.n_feature_buckets,
            index_hash_seed: rng.next_u32(),
            sign_hash_seed: rng.next_u32(),
        };
        let label_projector = HashingTrickProjector {
            n_buckets: self.n_label_buckets,
            index_hash_seed: rng.next_u32(),
            sign_hash_seed: rng.next_u32(),
        };
        TreeTrainer {
            feature_projector,
            label_projector,
            leaf_max_size: self.leaf_max_size,
            k_clusters: self.k_clusters,
            cluster_sample_size: self.cluster_sample_size,
            n_cluster_iters: self.n_cluster_iters,
            centroid_preserve_ratio: self.centroid_preserve_ratio,
            centroid_min_n_preserve: self.centroid_min_n_preserve,
        }
    }
}

struct TreeTrainer {
    feature_projector: HashingTrickProjector,
    label_projector: HashingTrickProjector,
    leaf_max_size: usize,
    k_clusters: u32,
    cluster_sample_size: usize,
    n_cluster_iters: u32,
    centroid_preserve_ratio: f32,
    centroid_min_n_preserve: usize,
}

impl TreeTrainer {
    fn train(&self, dataset: &DataSet) -> Tree {
        let root = {
            let feature_vectors = dataset
                .examples
                .iter()
                .map(|e| self.feature_projector.project_features(&e.features))
                .collect::<Vec<_>>();
            let label_vectors = dataset
                .examples
                .iter()
                .map(|e| self.label_projector.project_labels(&e.labels))
                .collect::<Vec<_>>();
            self.train_subtree(
                &feature_vectors.iter().collect::<Vec<_>>(),
                &label_vectors.iter().collect::<Vec<_>>(),
                &dataset
                    .examples
                    .iter()
                    .map(|e| &e.labels)
                    .collect::<Vec<_>>(),
            )
        };

        Tree {
            root,
            feature_projector: self.feature_projector.clone(),
        }
    }

    fn train_subtree(
        &self,
        feature_vectors: &[&SparseVector],
        label_vectors: &[&SparseVector],
        label_sets: &[&Vec<Label>],
    ) -> TreeNode {
        assert!(!feature_vectors.is_empty());
        assert_eq!(feature_vectors.len(), label_vectors.len());
        assert_eq!(label_vectors.len(), label_sets.len());

        // Return leaf if there aren't enough training examples left
        if feature_vectors.len() <= self.leaf_max_size {
            return Self::build_leaf(label_sets);
        }

        // Compute centroids and assign partitions accordingly
        let centroid_indices_pairs: Vec<(SparseVector, Vec<usize>)> = {
            let feature_centroids = self.train_node_classifier(feature_vectors, label_vectors);
            let mut partitions = vec![0; feature_vectors.len()];
            super::skmeans::reassign_partitions(
                feature_vectors,
                &feature_centroids,
                &mut partitions,
            );

            let mut indices_per_partition: Vec<Vec<usize>> =
                vec![Vec::new(); feature_centroids.len()];
            for (i, &p) in partitions.iter().enumerate() {
                let indices = &mut indices_per_partition[p];
                indices.push(i);
            }

            izip!(feature_centroids, indices_per_partition)
                .filter(|(_, v)| !v.is_empty())
                .collect()
        };
        assert!(!centroid_indices_pairs.is_empty());

        // If all examples are assigned to one centroid, also return leaf
        if centroid_indices_pairs.len() == 1 {
            let (_, indices) = &centroid_indices_pairs[0];
            assert!(indices.len() == feature_vectors.len());
            return Self::build_leaf(label_sets);
        }

        // Train a sub-tree for each partition
        let child_centroid_pairs = centroid_indices_pairs
            .into_par_iter()
            .map(|(centroid, indices)| {
                (
                    self.train_subtree(
                        &indices
                            .iter()
                            .map(|&i| feature_vectors[i])
                            .collect::<Vec<_>>(),
                        &indices
                            .iter()
                            .map(|&i| label_vectors[i])
                            .collect::<Vec<_>>(),
                        &indices.iter().map(|&i| label_sets[i]).collect::<Vec<_>>(),
                    ),
                    centroid,
                )
            }).collect();
        TreeNode::BranchNode {
            child_centroid_pairs,
        }
    }

    fn train_node_classifier_unsampled(
        &self,
        feature_vectors: &[&SparseVector],
        label_vectors: &[&SparseVector],
    ) -> Vec<SparseVector> {
        let (_, partitions) =
            super::skmeans::skmeans(label_vectors, self.k_clusters, self.n_cluster_iters);
        super::skmeans::compute_centroids_per_partition(
            feature_vectors,
            &partitions,
            self.centroid_preserve_ratio,
            self.centroid_min_n_preserve,
        )
    }

    fn train_node_classifier(
        &self,
        feature_vectors: &[&SparseVector],
        label_vectors: &[&SparseVector],
    ) -> Vec<SparseVector> {
        if feature_vectors.len() <= self.cluster_sample_size {
            // Data size smaller than desired sample size, skip sampling
            self.train_node_classifier_unsampled(feature_vectors, label_vectors)
        } else {
            let sampled_indices = rand::seq::index::sample(
                &mut thread_rng(),
                feature_vectors.len(),
                self.cluster_sample_size,
            );
            let mut sampled_feature_vectors = Vec::with_capacity(self.cluster_sample_size);
            let mut sampled_label_vectors = Vec::with_capacity(self.cluster_sample_size);
            for i in sampled_indices.iter() {
                sampled_feature_vectors.push(feature_vectors[i]);
                sampled_label_vectors.push(label_vectors[i]);
            }

            self.train_node_classifier_unsampled(&sampled_feature_vectors, &sampled_label_vectors)
        }
    }

    fn build_leaf(label_sets: &[&Vec<Label>]) -> TreeNode {
        TreeNode::LeafNode {
            label_score_pairs: aggregate_label_sets(label_sets),
        }
    }
}

fn aggregate_label_sets(label_sets: &[&Vec<Label>]) -> Vec<(Label, f32)> {
    let mut label_to_score = HashMap::new();

    for &label_set in label_sets {
        for label in label_set {
            let ref_score = label_to_score.entry(label.to_owned()).or_insert(0.);
            *ref_score += 1. / label_sets.len() as f32;
        }
    }

    label_to_score.into_iter().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_aggregate_label_sets() {
        use std::collections::HashMap;
        use std::iter::FromIterator;
        assert_eq!(
            hashmap![1 => 1., 2 => 2. / 3., 3 => 1. / 3.],
            HashMap::from_iter(
                super::aggregate_label_sets(&[&vec![1, 2, 3], &vec![1, 2], &vec![1],]).into_iter()
            ),
        );
    }
}
