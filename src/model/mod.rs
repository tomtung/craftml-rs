/// Spherical K-means clustering with K-means++ initialization.
mod skmeans;

/// Model training.
mod train;

/// Model testing & evaluation.
pub mod eval;

use bincode;
use data::{DataSet, Feature, Label, SparseVector};
use fasthash::murmur3::Murmur3Hasher_x86_32 as MurmurHasher;
use fasthash::FastHasher;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io;
use time;
use util::draw_async_progress_bar;

/// Trainer for CraftML models.
pub use self::train::CraftmlTrainer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CraftmlModel {
    trees: Vec<Tree>,
}

fn update_aggregate_with_tree_prediction(
    aggregate_prediction: &mut HashMap<Label, f32>,
    tree_prediction: &[(Label, f32)],
    n_trees: usize,
) {
    for &(ref label, score) in tree_prediction {
        let ref_score = aggregate_prediction.entry(label.to_owned()).or_insert(0.);
        *ref_score += score / n_trees as f32;
    }
}

fn aggregate_prediction_to_vec(aggregate_prediction: HashMap<Label, f32>) -> Vec<(Label, f32)> {
    let mut aggregate_prediction: Vec<_> = aggregate_prediction.into_iter().collect();
    aggregate_prediction
        .sort_unstable_by(|(_, score1), (_, score2)| score2.partial_cmp(score1).unwrap());
    aggregate_prediction
}

impl CraftmlModel {
    pub fn predict(&self, features: &[(Feature, f32)]) -> Vec<(Label, f32)> {
        let mut aggregate_prediction = HashMap::<Label, f32>::new();
        for tree_prediction in self.trees.iter().map(|tree| tree.predict(features)) {
            update_aggregate_with_tree_prediction(
                &mut aggregate_prediction,
                &tree_prediction,
                self.trees.len(),
            );
        }
        aggregate_prediction_to_vec(aggregate_prediction)
    }

    pub fn predict_all(&self, dataset: &DataSet) -> Vec<Vec<(Label, f32)>> {
        info!(
            "Calculating predictions for {} examples",
            dataset.examples.len()
        );
        let start_t = time::precise_time_s();

        let mut predictions = Vec::new();
        let (sender, handle) = draw_async_progress_bar(dataset.examples.len() as u64);
        dataset
            .examples
            .par_iter()
            .map_with(sender, |sender, e| {
                let prediction = self.predict(&e.features);
                sender.send(1).unwrap();
                prediction
            }).collect_into_vec(&mut predictions);

        let end_t = time::precise_time_s();
        handle.join().unwrap();
        info!(
            "Prediction on {} examples took {:.2}s",
            dataset.examples.len(),
            end_t - start_t,
        );

        predictions
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
    fn predict(&self, features: &[(Feature, f32)]) -> &Vec<(Label, f32)> {
        let feature_vector = self.feature_projector.project_features(features);
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
    fn hash(index: &impl Hash, seed: u32) -> u32 {
        let mut hasher = MurmurHasher::with_seed(seed);
        index.hash(&mut hasher);
        hasher.finish() as u32
    }

    fn hash_entry(&self, index: impl Hash, value: f32) -> (u16, f32) {
        let hashed_index = Self::hash(&index, self.index_hash_seed) as u16 % self.n_buckets;
        let sign = Self::hash(&index, self.sign_hash_seed) & 1 == 1;
        let value = if sign { -value } else { value };
        (hashed_index, value)
    }

    /// Project sparse features / labels with hashing trick.
    fn project<'a, S: 'a + Hash>(
        &self,
        index_value_pairs: impl Iterator<Item = (&'a S, f32)>,
    ) -> SparseVector {
        let mut index_to_value = HashMap::<u16, f32>::new();
        for (index, value) in index_value_pairs {
            let (index, value) = self.hash_entry(index, value);
            *index_to_value.entry(index).or_insert(0.) += value;
        }

        let mut sv = SparseVector::from(index_to_value);
        sv.l2_normalize();
        sv
    }

    fn project_features(&self, index_label_pairs: &[(Feature, f32)]) -> SparseVector {
        self.project(
            index_label_pairs
                .iter()
                .map(|(index, value)| (index, *value)),
        )
    }

    fn project_labels(&self, indices: &[Label]) -> SparseVector {
        self.project(indices.iter().map(|index| (index, 1.)))
    }
}
