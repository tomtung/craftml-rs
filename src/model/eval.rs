use super::{CraftmlModel, CraftmlTrainer};
use data::{DataSet, DataSplits, Label};
use std::cmp::min;
use std::collections::HashSet;
use std::iter::FromIterator;

/// Calculate precision@k metrics.
fn precision_at_k(
    max_k: usize,
    true_labels: &[&Vec<Label>],
    predicted_labels: &[Vec<(Label, f32)>],
) -> Vec<f32> {
    assert_eq!(true_labels.len(), predicted_labels.len());
    let mut ps = vec![0.; max_k];
    for (labels, predictions) in izip!(true_labels, predicted_labels) {
        let labels = HashSet::<Label>::from_iter(labels.iter().cloned());
        let mut n_correct = 0;
        for k in 0..min(max_k, predictions.len()) {
            if labels.contains(&predictions[k].0) {
                n_correct += 1;
            }
            ps[k] += n_correct as f32 / (k + 1) as f32;
        }
    }
    for p in &mut ps {
        *p /= true_labels.len() as f32;
    }
    ps
}

fn true_labels(dataset: &DataSet) -> Vec<&Vec<Label>> {
    dataset.examples.iter().map(|e| &e.labels).collect()
}

pub fn test_all(
    model: &CraftmlModel,
    test_dataset: &DataSet,
) -> (Vec<Vec<(Label, f32)>>, Vec<f32>) {
    let predicted_labels = model.predict_all(test_dataset);
    let precisions = precision_at_k(5, &true_labels(test_dataset), &predicted_labels);
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
