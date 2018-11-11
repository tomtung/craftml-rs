use super::{CraftmlModel, CraftmlTrainer};
use data::{DataSet, DataSplits, Label};
use rayon::prelude::*;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

/// Calculate precision@k metrics.
fn precision_at_k(
    max_k: usize,
    dataset: &DataSet,
    predicted_labels: &[Vec<(Label, f32)>],
) -> Vec<f32> {
    assert_eq!(dataset.examples.len(), predicted_labels.len());
    let mut ps = vec![0.; max_k];
    for (example, predictions) in izip!(&dataset.examples, predicted_labels) {
        let labels = HashSet::<Label>::from_iter(example.labels.iter().cloned());
        let mut n_correct = 0;
        for k in 0..min(max_k, predictions.len()) {
            if labels.contains(&predictions[k].0) {
                n_correct += 1;
            }
            ps[k] += n_correct as f32 / (k + 1) as f32;
        }
    }
    for p in &mut ps {
        *p /= predicted_labels.len() as f32;
    }
    ps
}

pub fn test_all(
    model: &CraftmlModel,
    test_dataset: &DataSet,
) -> (Vec<Vec<(Label, f32)>>, Vec<f32>) {
    let predicted_labels = model.predict_all(test_dataset);
    let precisions = precision_at_k(5, &test_dataset, &predicted_labels);
    info!(
        "Precision@[1, 3, 5] = [{:.2}, {:.2}, {:.2}]",
        precisions[0] * 100.,
        precisions[2] * 100.,
        precisions[4] * 100.,
    );
    (predicted_labels, precisions)
}

pub fn cross_validate(
    dataset: &DataSet,
    data_splits: &DataSplits,
    trainer: &CraftmlTrainer,
) -> Vec<f32> {
    let mut precisions = Vec::<f32>::new();
    for i in 0..data_splits.num_splits() {
        info!("Running cross validation with split #{}", i + 1);
        let (training_dataset, test_dataset) = data_splits.split_dataset(&dataset, i);
        let model = trainer.train(&training_dataset);
        let (_, split_precisions) = test_all(&model, &test_dataset);
        if i == 0 {
            precisions = split_precisions;
        } else {
            assert!(!precisions.is_empty());
            for (j, precision) in split_precisions.into_iter().enumerate() {
                precisions[j] += precision;
            }
        }
    }
    for precision in &mut precisions {
        *precision /= data_splits.num_splits() as f32;
    }
    info!(
        "Average precision@[1, 3, 5] = [{:.2}, {:.2}, {:.2}]",
        precisions[0] * 100.,
        precisions[2] * 100.,
        precisions[4] * 100.,
    );
    precisions
}

pub fn test_trees_singly(
    train_dataset: &DataSet,
    test_dataset: &DataSet,
    trainer: &CraftmlTrainer,
) -> (Vec<Vec<(Label, f32)>>, Vec<f32>) {
    let mut aggregate_predictions = vec![HashMap::<Label, f32>::new(); test_dataset.examples.len()];

    for i in 1..=trainer.n_trees {
        info!("Training tree #{}/{}", i, trainer.n_trees);
        let mut tree_trainer = trainer.clone();
        tree_trainer.n_trees = 1;
        let model = tree_trainer.train(train_dataset);
        let tree_predictions = model.predict_all(test_dataset);

        (&mut aggregate_predictions)
            .into_par_iter()
            .zip((&tree_predictions).into_par_iter())
            .for_each(|(aggregate_prediction, tree_prediction)| {
                super::update_aggregate_with_tree_prediction(
                    aggregate_prediction,
                    tree_prediction,
                    trainer.n_trees,
                );
            });
    }

    let aggregate_predictions: Vec<_> = aggregate_predictions
        .into_par_iter()
        .map(super::aggregate_prediction_to_vec)
        .collect();
    let precisions = precision_at_k(5, test_dataset, &aggregate_predictions);

    info!(
        "Aggregate precision@[1, 3, 5] = [{:.2}, {:.2}, {:.2}]",
        precisions[0] * 100.,
        precisions[2] * 100.,
        precisions[4] * 100.,
    );

    (aggregate_predictions, precisions)
}
