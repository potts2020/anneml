/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use arrayfire::Array;
use rustc_hash::FxHashMap;
use crate::node::edges::NodeRange;

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub(crate) struct Tensor{
    pub(crate) hash_map: FxHashMap<String, Array<f64>>,
}

impl Tensor {
    pub(crate) fn new(key_value_pair: &[(&str, Array<f64>)]) -> Tensor {
        let mut hash_map = FxHashMap::default();
        for (k, v) in key_value_pair { hash_map.insert(k.to_string(), v.to_owned()); }
        Tensor { hash_map }
    }

    pub(crate) fn insert(&mut self, key: &str, value: Array<f64>) {
        self.hash_map.insert(key.to_string(), value);
    }

    pub(crate) fn merge(&mut self, tensor: &Tensor, node_range: &NodeRange) {
        tensor.hash_map.iter().for_each(|(k,v)|
            match node_range {
                NodeRange::Selective(key_chain) => { if key_chain.contains(k) { self.hash_map.insert(k.to_string(), v.to_owned()); } }
                NodeRange::All => { self.hash_map.insert(k.to_string(), v.to_owned()); }
            }
        );
    }
}
