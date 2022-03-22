/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use rustc_hash::{FxHasher, FxHashMap};
use crate::node::scope::Scope;

#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct Attribute {
    activation: Activation,
    cell_type: CellType,
    descriptor: FxHashMap<String, TensorDescriptor>,
    scope: Scope,
}

impl Attribute {

    /// Create an Attribute that is the blueprint to Node behaviours.
    ///
    /// An Attribute define how nodes are constructed and processed.
    /// Attributes can be assigned on a to layers within a Node, with each layer having a different Attribute.
    ///
    /// # Example(s)
    /// ```
    /// // Create a Attribute
    /// use anneml::node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
    /// use anneml::node::scope::Scope;
    ///
    /// let activation = Activation::Sigmoid;
    /// let cell_type = CellType::Mlp;
    /// let descriptor = vec![("WEIGHTS", TensorDescriptor::RandN),("BIASES", TensorDescriptor::Const(1.3))];
    /// let scope = Scope::new(1,1);
    ///
    /// let attribute = Attribute::new(activation, cell_type, descriptor, scope);
    /// ```
    pub fn new(activation: Activation, cell_type: CellType, descriptor: Vec<(&str, TensorDescriptor)>, scope: Scope) -> Attribute {
        Attribute { activation,
            cell_type, descriptor: Attribute::build_descriptors(descriptor), scope,
        }
    }

    /// Maps a vector of (&str,TensorDescriptor) key value pairs to a hashmap.
    ///
    /// Descriptions are hashmaps that describe how to seed values in an individual layer.
    fn build_descriptors(descriptor: Vec<(&str, TensorDescriptor)>) -> HashMap<String, TensorDescriptor, BuildHasherDefault<FxHasher>> {
        let mut op_seq = FxHashMap::default();
        descriptor.into_iter().for_each(|(key, value)| {op_seq.insert(key.to_string(), value); });
        op_seq
    }

    /// Gets a TensorDescriptor reference associated with the description key.
    pub fn description(&self, key: &str ) -> &TensorDescriptor {
        self.descriptor.get(key).unwrap()
    }

    /// Acquire an Activation reference associated with the Attribute.
    pub fn activation(&self) -> &Activation{
        &self.activation
    }

    /// Acquire an CellType reference associated with the Attribute.
    pub fn cell_type(&self) -> &CellType{
        &self.cell_type
    }

    /// Acquire an Scope reference associated with the Attribute.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum Activation{
    Sigmoid,
    TanH,
    None,
}

impl Default for Activation {
    fn default() -> Self {
        Activation::Sigmoid
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum CellType {
    Mlp,
}

impl Default for CellType {
    fn default() -> Self {
        CellType::Mlp
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum TensorDescriptor {
    RandN,
    RandU,
    RangeN(f64),
    RangeU(f64),
    Range(f64),
    Const(f64),
}

impl Default for TensorDescriptor {
    fn default() -> Self {
        TensorDescriptor::RandN
    }
}