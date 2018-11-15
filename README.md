# craftml-rs
A Rust implementation of CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning (Siblini et al., 2018).

## Performance

This implementation has been tested on datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). Each data set comes either with a single data file and separate files for train / test splits, or with two separate train / test data files.

A data file starts with a header line with three space-separated integers: total number of examples, number of features, and number of labels. Following the header line, there is one line per each example, starting with comma-separated labels, followed by space-separated feature:value pairs:
```
label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
```

A split file is a integer matrix, with one line per row, and columns separated by spaces. The integers are example indices (1-indexed) in the corresponding data file, and each column corresponds to a separate split.

Precisions at 1, 3, and 5 are calculated for models trained with default hyper-parameters, e.g.
- `craftml train Mediamill/data.txt --cv_splits_path Mediamill/train_split.txt` for Mediamill, which has a single data file and separate train / test split files;
- `craftml train EURLex-4K/train.txt --test_data EURLex-4K/test.txt` for EURLex-4K, which has separate train / test data files.

| Dataset | P@1 | P@3 | P@5 |
| --- | --- | --- | --- |
| Mediamill | 85.51 | 69.94 | 56.39 |
| Bibtex | 61.47 | 37.20 | 27.32 |
| Delicious | 67.78 | 62.15 | 57.63 |
| EURLex-4K | 79.52 | 66.42 | 55.25 |
| Wiki10-31K | 83.57 | 72.69 | 63.65 |
| WikiLSHTC-325K | 51.79 | 32.41 | 23.43 |
| Delicious-200K | 47.34 | 40.85 | 37.67 |
| Amazon-670K | 38.40 | 34.21 | 31.41 |
| AmazonCat-13K | 92.88 | 77.48 | 61.32 |

These numbers are generally consistent with those reported in the original paper.

Note that if there isn't enough memory to train on a large data set, the `--test_trees_singly` flag can be set to only train & test one tree at a time, and discard each tree when it's been tested. This allows one to obtain test results without being able to fit the entire model in memory. One can also tune the `--centroid_preserve_ratio` option to trade off between model size and accuracy.

## Build
The project can be easily built with [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html):
```
$ cargo build --release
```

The compiled binary file will be available at `target/release/craftml`.

## Usage
```
$ craftml train --help

craftml-train
Train a new CRAFTML model

USAGE:
    craftml train [FLAGS] [OPTIONS] <training_data>

FLAGS:
    -h, --help                 Prints help information
        --test_trees_singly    Test forest tree by tree, freeing each before training the next to reduce memory usage.
                               Model cannot be saved.
    -V, --version              Prints version information

OPTIONS:
        --centroid_min_n_preserve <centroid_min_n_preserve>
            The minimum number of entries to preserve from puning, regardless preserve ratio setting. [default: 10]

        --centroid_preserve_ratio <centroid_preserve_ratio>
            A real number between 0 and 1, which is the ratio of entries with largest absoulte values to preserve. The
            rest of the entries are pruned. [default: 0.1]
        --cluster_sample_size <cluster_sample_size>
            Number of examples drawn for clustering on a branching node [default: 20000]

        --cv_splits_path <PATH>
            Path to the k-fold cross validation splits file, with k space-separated columns of indices (starting from 1)
            for training splits.
        --k_clusters <k_clusters>                              Number of clusters on a branching node [default: 10]
        --leaf_max_size <leaf_max_size>
            Maximum number of distinct examples on a leaf node [default: 10]

        --model_path <PATH>                                    Path to which the trained model will be saved if provided
        --n_cluster_iters <n_cluster_iters>
            Number of clustering iterations to run on each branching node [default: 2]

        --n_feature_buckets <n_feature_buckets>
            Number of buckets into which features are hashed [default: 10000]

        --n_label_buckets <n_label_buckets>
            Number of buckets into which labels are hashed [default: 10000]

        --n_threads <n_threads>
            Number of worker threads. If 0, the number is selected automatically. [default: 0]

        --n_trees <n_trees>                                    Number of trees in the random forest [default: 50]
        --out_path <PATH>
            Path to the which predictions will be written, if provided

        --test_data <PATH>
            Path to test dataset file used to calculate metrics if provided (in the format of the Extreme Classification
            Repository)

ARGS:
    <training_data>    Path to training dataset file (in the format of the Extreme Classification Repository)
```

```
$ craftml test --help

craftml-test
Test an existing CRAFTML model

USAGE:
    craftml test [OPTIONS] <model_path> <test_data>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --k_top <k_top>            Number of top predictions to write out for each test example [default: 5]
        --n_threads <n_threads>    Number of worker threads. If 0, the number is selected automatically. [default: 0]
        --out_path <PATH>          Path to the which predictions will be written, if provided

ARGS:
    <model_path>    Path to the trained model
    <test_data>     Path to test dataset file (in the format of the Extreme Classification Repository)
```

## References

- Siblini, W., Kuntz, P., & Meyer, F. (2018). *CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning.* In Proceedings of the 35th International Conference on Machine Learning (Vol. 80, pp. 4664–4673). Stockholmsmässan, Stockholm Sweden: PMLR. http://proceedings.mlr.press/v80/siblini18a.html
