use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use rustc_hash::{FxHasher, FxHashMap};
use crate::mod_node::scope::Scope;

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub (crate) struct Attribute {
    activation: Activation,
    node_type: CellType,
    descriptor: FxHashMap<String, TensorDescriptor>,
    scope: Scope,
}

impl Attribute {
    pub(crate) fn new(activation: Activation, node_type: CellType, descriptor: Vec<(&str, TensorDescriptor)>, scope: Scope) -> Attribute {
        Attribute { activation, node_type, descriptor: Attribute::build_descriptions(descriptor), scope,
        }
    }

    fn build_descriptions(descriptor: Vec<(&str, TensorDescriptor)>) -> HashMap<String, TensorDescriptor, BuildHasherDefault<FxHasher>> {
        let mut op_seq = FxHashMap::default();
        descriptor.into_iter().for_each(|(key, value)| {op_seq.insert(key.to_string(), value); });
        op_seq
    }

    pub(crate) fn descriptions(&self, key: &str ) -> &TensorDescriptor {
        self.descriptor.get(key).unwrap()
    }

    pub(crate) fn activation(&self) -> &Activation{
        &self.activation
    }

    pub(crate) fn cell_type(&self) -> &CellType{
        &self.node_type
    }

    pub(crate) fn scope(&self) -> &Scope {
        &self.scope
    }
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(crate) enum Activation{
    Sigmoid,
    TanH,
    None,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(crate) enum CellType {
    Mlp,
}

/// Tensor Operation enumeration determines the Tensor operation to apple to a Arrayfire Array.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(crate) enum TensorDescriptor {
    RandN,
    RandU,
    RangeN(f64),
    RangeU(f64),
    Range(f64),
    Const(f64),
}