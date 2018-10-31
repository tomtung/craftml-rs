use super::data;
use super::util::draw_async_progress_bar;
use bincode;
use data::{DataSet, Feature, Label, SparseVector};
use fasthash::murmur3::hash32_with_seed;
use rand;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io;
use time;

/// Spherical K-means clustering with K-means++ initialization.
mod skmeans;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CraftmlModel {
    trees: Vec<Tree>,
}

impl CraftmlModel {
    pub fn predict(&self, features: &HashMap<Feature, f32>) -> Vec<(Label, f32)> {
        let mut label_score_pairs: Vec<_> = {
            let mut aggregate_map = HashMap::<Label, f32>::new();

            for tree_prediction in self.trees.iter().map(|tree| tree.predict(features)) {
                for &(ref label, score) in tree_prediction {
                    let ref_score = aggregate_map.entry(label.to_owned()).or_insert(0.);
                    *ref_score += score / self.trees.len() as f32;
                }
            }

            aggregate_map.into_iter().collect()
        };
        label_score_pairs
            .sort_unstable_by(|(_, score1), (_, score2)| score2.partial_cmp(score1).unwrap());
        label_score_pairs
    }

    pub fn save<W: io::Write>(&self, writer: W) -> io::Result<()> {
        info!("Saving model...");
        let start_t = time::precise_time_s();

        bincode::serialize_into(writer, self)
            .or_else(|e| Err(io::Error::new(io::ErrorKind::Other, e)))?;

        info!(
            "Model saved; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        Ok(())
    }

    pub fn load<R: io::Read>(reader: R) -> io::Result<Self> {
        info!("Loading model...");
        let start_t = time::precise_time_s();

        let model: Self = bincode::deserialize_from(reader)
            .or_else(|e| Err(io::Error::new(io::ErrorKind::Other, e)))?;
        info!(
            "Model loaded; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        Ok(model)
    }
}

#[derive(Debug)]
pub struct CraftmlTrainer {
    pub n_trees: usize,
    pub n_feature_buckets: u16,
    pub n_label_buckets: u16,
    pub leaf_max_size: usize,
    pub k_clusters: u32,
    pub cluster_sample_size: usize,
    pub n_cluster_iters: u32,
    pub centroid_min_value: f32,
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
            centroid_min_value: 0.01,
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
        assert!(self.centroid_min_value >= 0.);

        info!("Training CRAFTML model with parameters {:?}", self);
        let sender = draw_async_progress_bar(self.n_trees as u64);
        let start_t = time::precise_time_s();

        let mut trees = Vec::new();
        (0..self.n_trees)
            .into_par_iter()
            .map_with(sender, |sender, _| {
                let tree_trainer = TreeTrainer::new(
                    dataset.n_features.min(self.n_feature_buckets as u32) as u16,
                    dataset.n_labels.min(self.n_label_buckets as u32) as u16,
                    self.leaf_max_size,
                    self.k_clusters,
                    self.cluster_sample_size,
                    self.n_cluster_iters,
                    self.centroid_min_value,
                );
                let tree = tree_trainer.train(dataset);
                sender.send(1).unwrap();
                tree
            }).collect_into_vec(&mut trees);

        info!(
            "CRAFTML model training complete; it took {:.2}s",
            time::precise_time_s() - start_t
        );
        CraftmlModel { trees }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Tree {
    root: TreeNode,
    feature_projector: HashingTrickProjector,
}

impl Tree {
    fn predict(&self, features: &HashMap<Feature, f32>) -> &Vec<(Label, f32)> {
        let feature_vector = self.feature_projector.project_map(features);
        self.root.predict(&feature_vector)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum TreeNode {
    LeafNode {
        label_score_pairs: Vec<(Label, f32)>,
    },
    BranchNode {
        child_centroid_pairs: Vec<(TreeNode, SparseVector)>,
    },
}

impl TreeNode {
    fn predict(&self, feature_vector: &SparseVector) -> &Vec<(Label, f32)> {
        match self {
            TreeNode::LeafNode {
                ref label_score_pairs,
            } => label_score_pairs,
            TreeNode::BranchNode {
                ref child_centroid_pairs,
            } => {
                let p = skmeans::assign_partition(
                    feature_vector,
                    child_centroid_pairs.iter().map(|&(_, ref c)| c),
                );
                child_centroid_pairs[p].0.predict(feature_vector)
            }
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
    centroid_min_value: f32,
}

impl TreeTrainer {
    fn new(
        n_feature_buckets: u16,
        n_label_buckets: u16,
        leaf_max_size: usize,
        k_clusters: u32,
        cluster_sample_size: usize,
        n_cluster_iters: u32,
        centroid_min_value: f32,
    ) -> TreeTrainer {
        let mut rng = thread_rng();
        let feature_projector = HashingTrickProjector {
            n_buckets: n_feature_buckets,
            index_hash_seed: rng.next_u32(),
            sign_hash_seed: rng.next_u32(),
        };
        let label_projector = HashingTrickProjector {
            n_buckets: n_label_buckets,
            index_hash_seed: rng.next_u32(),
            sign_hash_seed: rng.next_u32(),
        };
        TreeTrainer {
            feature_projector,
            label_projector,
            leaf_max_size,
            k_clusters,
            cluster_sample_size,
            n_cluster_iters,
            centroid_min_value,
        }
    }

    fn train(&self, dataset: &DataSet) -> Tree {
        let root = {
            let feature_vectors = dataset
                .features_per_example
                .iter()
                .map(|features| self.feature_projector.project_map(features))
                .collect::<Vec<_>>();
            let label_vectors = dataset
                .labels_per_example
                .iter()
                .map(|labels| self.label_projector.project_set(labels))
                .collect::<Vec<_>>();
            self.train_subtree(
                &feature_vectors.iter().collect::<Vec<_>>(),
                &label_vectors.iter().collect::<Vec<_>>(),
                &dataset.labels_per_example.iter().collect::<Vec<_>>(),
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
        label_sets: &[&HashSet<Label>],
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
            skmeans::reassign_partitions(feature_vectors, &feature_centroids, &mut partitions);

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
        assert!(centroid_indices_pairs.len() > 0);

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
            skmeans::skmeans(label_vectors, self.k_clusters, self.n_cluster_iters);
        skmeans::compute_centroids_per_partition(
            feature_vectors,
            &partitions,
            self.centroid_min_value,
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

    fn build_leaf(label_sets: &[&HashSet<Label>]) -> TreeNode {
        TreeNode::LeafNode {
            label_score_pairs: aggregate_label_sets(label_sets),
        }
    }
}

fn aggregate_label_sets(label_sets: &[&HashSet<Label>]) -> Vec<(Label, f32)> {
    let mut label_to_score = HashMap::new();

    for &label_set in label_sets {
        for label in label_set {
            let ref_score = label_to_score.entry(label.to_owned()).or_insert(0.);
            *ref_score += 1. / label_sets.len() as f32;
        }
    }

    label_to_score.into_iter().collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HashingTrickProjector {
    n_buckets: u16,
    index_hash_seed: u32,
    sign_hash_seed: u32,
}

impl HashingTrickProjector {
    /// Project sparse features / labels with hashing trick.
    fn project<'a, T, S: 'a>(&self, index_value_pairs: T) -> SparseVector
    where
        T: Iterator<Item = (&'a S, f32)>,
        S: AsRef<[u8]>,
    {
        let mut index_to_value = HashMap::<u16, f32>::new();
        for (feature, value) in index_value_pairs {
            let ref_v = {
                let index =
                    hash32_with_seed(&feature, self.index_hash_seed) as u16 % self.n_buckets;
                index_to_value.entry(index).or_insert(0.)
            };
            *ref_v += {
                let sign = hash32_with_seed(&feature, self.sign_hash_seed) & 1 == 1;
                if sign {
                    -value
                } else {
                    value
                }
            };
        }

        let mut sv = SparseVector::from(index_to_value);
        sv.l2_normalize();
        sv
    }

    fn project_map(&self, index_to_value: &HashMap<String, f32>) -> SparseVector {
        self.project(index_to_value.iter().map(|(index, &value)| (index, value)))
    }

    fn project_set(&self, indices: &HashSet<String>) -> SparseVector {
        self.project(indices.iter().map(|index| (index, 1.)))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_aggregate_label_sets() {
        assert_eq!(
            hashmap!{
                "1".to_owned() => 1.,
                "2".to_owned() => 2. / 3.,
                "3".to_owned() => 1. / 3.,
            },
            super::aggregate_label_sets(&[
                &hashset!{"1".to_owned(), "2".to_owned(), "3".to_owned()},
                &hashset!{"1".to_owned(), "2".to_owned()},
                &hashset!{"1".to_owned()},
            ]).into_iter()
            .collect(),
        );
    }
}
