extern crate craftml;
#[macro_use]
extern crate clap;
extern crate rayon;

use craftml::data::{DataSet, DataSplits, Label};
use craftml::model::{eval, CraftmlModel, CraftmlTrainer};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

fn set_num_threads(arg_matches: &clap::ArgMatches) {
    let n_threads = arg_matches
        .value_of("n_threads")
        .and_then(|s| s.parse::<usize>().ok())
        .expect("Failed to parse n_threads");

    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build_global()
        .unwrap();
}

macro_rules! parse_trainer {
    ($m:ident; $( $v:ident ),+) => {{
        let mut trainer = CraftmlTrainer::default();
        $(
            if let Some($v) = $m.value_of(stringify!($v)) {
                trainer.$v = $v.parse().expect(&format!("Failed to parse {}", stringify!($v)));
            }
        )*
        trainer
    }};
}

fn maybe_write_predictions_file(arg_matches: &clap::ArgMatches, predictions: &[Vec<(Label, f32)>]) {
    if let Some(out_path) = arg_matches.value_of("out_path") {
        let k_top = arg_matches
            .value_of("k_top")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Failed to parse k_top");

        let mut writer =
            BufWriter::new(File::create(out_path).expect("Failed to create output file"));
        for prediction in predictions {
            for (i, &(ref label, score)) in prediction.iter().take(k_top).enumerate() {
                if i > 0 {
                    write!(&mut writer, "\t");
                }
                write!(&mut writer, "{} {:.3}", label, score);
            }
            writeln!(&mut writer);
        }
    }
}

fn train(arg_matches: &clap::ArgMatches) {
    set_num_threads(&arg_matches);
    let trainer = parse_trainer!(arg_matches;
            n_trees, n_feature_buckets, n_label_buckets, leaf_max_size, k_clusters,
            cluster_sample_size, n_cluster_iters, centroid_preserve_ratio, centroid_min_n_preserve);

    let training_path = arg_matches.value_of("training_data").unwrap();
    let training_dataset =
        DataSet::load_xc_repo_data_file(training_path).expect("Failed to load training data");

    if let Some(cv_splits_path) = arg_matches.value_of("cv_splits_path") {
        let data_splits = DataSplits::parse_xc_repo_data_split_file(cv_splits_path)
            .expect("Failed to load splits");
        eval::cross_validate(&training_dataset, &data_splits, &trainer);
    } else if arg_matches.is_present("test_trees_singly") {
        let test_path = arg_matches.value_of("test_data").unwrap();
        let test_dataset =
            DataSet::load_xc_repo_data_file(test_path).expect("Failed to load test data");

        let (predictions, _) = eval::test_trees_singly(&training_dataset, &test_dataset, &trainer);
        maybe_write_predictions_file(arg_matches, &predictions);
    } else {
        let model = trainer.train(&training_dataset);

        if let Some(model_path) = arg_matches.value_of("model_path") {
            let model_file = File::create(model_path).expect("Failed to create model file");
            model
                .save(BufWriter::new(model_file))
                .expect("Failed to save model");
        }

        if let Some(test_path) = arg_matches.value_of("test_data") {
            let test_dataset =
                DataSet::load_xc_repo_data_file(test_path).expect("Failed to load test data");
            let (predictions, _) = eval::test_all(&model, &test_dataset);
            maybe_write_predictions_file(arg_matches, &predictions);
        }
    }
}

fn test(arg_matches: &clap::ArgMatches) {
    set_num_threads(&arg_matches);
    let model_path = arg_matches.value_of("model_path").unwrap();
    let model_file = File::open(model_path).expect("Failed to open model file");
    let model = CraftmlModel::load(BufReader::new(model_file)).expect("Failed to load model");

    let test_path = arg_matches.value_of("test_data").unwrap();
    let test_dataset =
        DataSet::load_xc_repo_data_file(test_path).expect("Failed to load test data");
    let (predictions, _) = eval::test_all(&model, &test_dataset);
    maybe_write_predictions_file(arg_matches, &predictions);
}

fn main() {
    simple_logger::init().unwrap();

    let yaml = load_yaml!("cli.yml");
    let arg_matches = clap::App::from_yaml(yaml).get_matches();

    if let Some(arg_matches) = arg_matches.subcommand_matches("train") {
        train(&arg_matches);
    } else if let Some(arg_matches) = arg_matches.subcommand_matches("test") {
        test(&arg_matches);
    } else {
        println!("{}", arg_matches.usage());
    }
}
