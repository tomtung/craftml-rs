# craftml-rs
A Rust implementation of CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning (Siblini et al., 2018).

## Performance

This implementation has been tested on datasets from the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). Precisions at 1, 3, and 5 are calculated for models trained with default hyper-parameters (e.g. to train the model for EURLex-4K we simply run `craftml train eurlex_train.txt --test_data eurlex_test.txt` without changing any settings).

| Dataset | P@1 | P@3 | P@5 |
| --- | --- | --- | --- |
| Mediamill | 86.67 | 71.53 | 57.59 |
| Bibtex | 62.39 | 37.93 | 27.41 |
| Delicious | 69.02 | 63.37 | 58.66 |
| EURLex-4K | 79.1 | 65.24 | 53.97 |
| Wiki10-31K | 85.1 | 73.65 | 63.99 |

These numbers are generally consistent with those reported in the original paper.

_TODO(?): My PC doesn't have enough memory to train on the larger datasets._

## Build
The project can be easily built with [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html):
```
$ cargo build --release
```

The compiled binary file will be available at `target/release/craftml`.

## Usage
```
$ target/release/craftml train --help
craftml-train
Train a new CRAFTML model

USAGE:
    craftml train [OPTIONS] <training_data> --model_path <PATH> --test_data <PATH>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --cluster_sample_size <cluster_sample_size>
            Number of examples drawn for clustering on a branching node [default: 20000]

        --k_clusters <k_clusters>                      Number of clusters on a branching node [default: 10]
        --leaf_max_size <leaf_max_size>                Maximum number of distinct examples on a leaf node [default: 10]
        --model_path <PATH>                            Path to which the trained model will be saved if provided
        --n_cluster_iters <n_cluster_iters>
            Number of clustering iterations to run on each branching node [default: 2]

        --n_feature_buckets <n_feature_buckets>        Number of buckets into which features are hashed [default: 10000]
        --n_label_buckets <n_label_buckets>            Number of buckets into which labels are hashed [default: 10000]
        --n_threads <n_threads>
            Number of worker threads. If 0, the number is selected automatically. [default: 0]

        --n_trees <n_trees>                            Number of trees in the random forest [default: 50]
        --test_data <PATH>
            Path to test dataset file used to calculate metrics if provided (in the format of the Extreme Classification
            Repository)

ARGS:
    <training_data>    Path to training dataset file (in the format of the Extreme Classification Repository)
```

```
$ target/release/craftml test --help
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

The program accepts dataset files in the [Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html) format as inputs.

## References

- Siblini, W., Kuntz, P., & Meyer, F. (2018). *CRAFTML, an Efficient Clustering-based Random Forest for Extreme Multi-label Learning.* In Proceedings of the 35th International Conference on Machine Learning (Vol. 80, pp. 4664–4673). Stockholmsmässan, Stockholm Sweden: PMLR. http://proceedings.mlr.press/v80/siblini18a.html
