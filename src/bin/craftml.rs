extern crate craftml;
extern crate time;
#[macro_use]
extern crate log;
extern crate pbr;
extern crate simple_logger;
#[macro_use]
extern crate clap;
extern crate rayon;

use craftml::data::DataSet;
use craftml::metrics::precision_at_k;
use craftml::model::{CraftmlModel, CraftmlTrainer};
use craftml::util::draw_async_progress_bar;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

fn predict_all(model: &CraftmlModel, test_dataset: &DataSet) -> Vec<Vec<(String, f32)>> {
    info!(
        "Calculating predictions for {} test examples",
        test_dataset.features_per_example.len()
    );
    let start_t = time::precise_time_s();

    let mut predictions = Vec::new();
    let sender = draw_async_progress_bar(test_dataset.features_per_example.len() as u64);
    test_dataset
        .features_per_example
        .par_iter()
        .map_with(sender, |sender, feature_map| {
            let prediction = model.predict(feature_map);
            sender.send(1).unwrap();
            prediction
        }).collect_into_vec(&mut predictions);

    let end_t = time::precise_time_s();
    let precisions = precision_at_k(5, &test_dataset.labels_per_example, &predictions);
    info!(
        "Prediction on {} examples took {:.2}s; precision@[1, 3, 5] = [{:.2}, {:.2}, {:.2}]",
        test_dataset.n_examples,
        end_t - start_t,
        precisions[0] * 100.,
        precisions[2] * 100.,
        precisions[4] * 100.,
    );

    predictions
}

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

fn train(arg_matches: &clap::ArgMatches) {
    set_num_threads(&arg_matches);
    let trainer = parse_trainer!(arg_matches;
            n_trees, n_feature_buckets, n_label_buckets, leaf_max_size, k_clusters,
            cluster_sample_size, n_cluster_iters, centroid_preserve_ratio, centroid_min_n_preserve);

    let training_path = arg_matches.value_of("training_data").unwrap();
    let training_dataset = DataSet::load_xc_repo_data_file(training_path).unwrap();

    let model = trainer.train(&training_dataset);

    if let Some(model_path) = arg_matches.value_of("model_path") {
        let model_file = File::create(model_path).unwrap();
        model.save(BufWriter::new(model_file)).unwrap();
    }

    if let Some(test_path) = arg_matches.value_of("test_data") {
        let test_dataset = DataSet::load_xc_repo_data_file(test_path).unwrap();
        predict_all(&model, &test_dataset);
    }
}

fn test(arg_matches: &clap::ArgMatches) {
    set_num_threads(&arg_matches);
    let model_path = arg_matches.value_of("model_path").unwrap();
    let model_file = File::open(model_path).unwrap();
    let model = CraftmlModel::load(BufReader::new(model_file)).unwrap();

    let test_path = arg_matches.value_of("test_data").unwrap();
    let test_dataset = DataSet::load_xc_repo_data_file(test_path).unwrap();
    let predictions = predict_all(&model, &test_dataset);

    if let Some(out_path) = arg_matches.value_of("out_path") {
        let k_top = arg_matches
            .value_of("k_top")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Failed to parse k_top");

        let mut writer = BufWriter::new(File::create(out_path).unwrap());
        for prediction in &predictions {
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
