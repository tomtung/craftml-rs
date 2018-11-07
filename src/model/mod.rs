/// Spherical K-means clustering with K-means++ initialization.
mod skmeans;

/// Model training.
mod train;

use bincode;
use data::{Feature, Label, SparseVector};
use fasthash::murmur3::hash32_with_seed;
use std::collections::{HashMap, HashSet};
use std::io;
use time;

/// Trainer for CraftML models.
pub use self::train::CraftmlTrainer;

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
