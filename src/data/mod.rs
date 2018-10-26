use pbr::ProgressBar;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, Error, ErrorKind, Result};
use time;

mod sparse_vector;
pub use self::sparse_vector::SparseVector;

pub type Feature = String;

pub type Label = String;

pub struct DataSet {
    pub n_examples: u32,
    pub n_features: u32,
    pub n_labels: u32,
    pub features_per_example: Vec<HashMap<Feature, f32>>,
    pub labels_per_example: Vec<HashSet<Label>>,
}

impl DataSet {
    /// Parse a line in a data file from the Extreme Classification Repository
    ///
    /// The line should be in the following format:
    /// label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
    ///
    /// Here we treat feature and label indices as strings for flexibility.
    fn parse_xc_repo_data_line(line: &str) -> Result<(HashMap<Feature, f32>, HashSet<Label>)> {
        let mut token_iter = line.split(' ');
        let labels = token_iter
            .next()
            .unwrap()
            .split(',')
            .filter_map(|s| {
                if s.is_empty() {
                    None
                } else {
                    Some(s.to_string())
                }
            }).collect();

        let mut feature_to_value = HashMap::new();
        for feature_value_pair_str in token_iter {
            let mut feature_value_pair_iter = feature_value_pair_str.split(':');
            let feature = feature_value_pair_iter
                .next()
                .ok_or(ErrorKind::InvalidData)?
                .to_string();
            let value = feature_value_pair_iter
                .next()
                .and_then(|s| s.parse::<f32>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            if feature_value_pair_iter.next().is_some() {
                return Err(Error::from(ErrorKind::InvalidData));
            }
            feature_to_value.insert(feature, value);
        }

        Ok((feature_to_value, labels))
    }

    /// Load a data file from the Extreme Classification Repository
    pub fn load_xc_repo_data_file(path: &str) -> Result<Self> {
        info!("Loading data from {}", path);
        let start_t = time::precise_time_s();

        let mut lines = BufReader::new(File::open(path)?).lines();

        let (n_examples, n_features, n_labels) = {
            let header_line = lines.next().ok_or(ErrorKind::InvalidData)??;
            let mut token_iter = header_line.split_whitespace();
            let n_examples = token_iter
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            let n_features = token_iter
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            let n_labels = token_iter
                .next()
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or(ErrorKind::InvalidData)?;
            if token_iter.next().is_some() {
                Err(ErrorKind::InvalidData)?;
            }

            (n_examples, n_features, n_labels)
        };

        let mut features_per_example = Vec::new();
        let mut labels_per_example = Vec::new();
        let mut pb = ProgressBar::on(::std::io::stderr(), n_examples.into());
        for line in lines {
            let (feature_map, label_set) = Self::parse_xc_repo_data_line(&line?)?;
            features_per_example.push(feature_map);
            labels_per_example.push(label_set);
            pb.inc();
        }
        assert_eq!(features_per_example.len(), labels_per_example.len());
        if n_examples as usize != features_per_example.len() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected {} examples, only read {} lines",
                    n_examples,
                    features_per_example.len()
                ),
            ));
        }

        info!(
            "Loaded {} examples; it took {:.2}s",
            n_examples,
            time::precise_time_s() - start_t
        );
        Ok(Self {
            n_examples,
            n_features,
            n_labels,
            features_per_example,
            labels_per_example,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parse_xc_repo_data_line() {
        assert_eq!(
            (
                hashmap! {
                    "feature1".to_owned()=> 1.,
                    "feature2".to_owned()=> 2.,
                    "feature3".to_owned()=> 3.,
                },
                hashset!{"label1".to_owned(), "label2".to_owned()}
            ),
            super::DataSet::parse_xc_repo_data_line(
                "label1,label2 feature1:1 feature2:2 feature3:3"
            ).unwrap()
        );
    }
}
