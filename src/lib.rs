#[macro_use]
extern crate itertools;

extern crate rand;

#[macro_use]
extern crate serde_derive;

#[cfg(test)]
#[macro_use]
extern crate maplit;

#[macro_use]
extern crate log;

extern crate simple_logger;

extern crate fasthash;

extern crate time;

extern crate pbr;

extern crate bincode;

extern crate rayon;

extern crate order_stat;

/// Data representations and processing.
pub mod data;

/// Model implementation.
pub mod model;

pub mod util;
