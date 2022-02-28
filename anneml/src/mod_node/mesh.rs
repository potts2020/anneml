use std::sync::{Mutex, MutexGuard};
use crate::mod_node::edges::Edges;
use crate::mod_node::layer::Layer;
use crate::mod_node::tensor::Tensor;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Mesh {
    layers: Vec<Mutex<Layer>>,
    tensor: Mutex<Tensor>,
    edges:  Mutex<Edges>,
}

impl Mesh {
    pub(crate) fn new(layers: Vec<Mutex<Layer>>, edges: Mutex<Edges>) -> Self {
        Mesh { layers, tensor: Mutex::new(Default::default()), edges
        }
    }

    pub(crate) fn topology(&self) -> Vec<usize> {
        self.layers().iter().map(|e| e.lock().unwrap().tensor.hash_map
            .iter().filter(|(k,_)| !k.contains("_SYSTEM")).map(|(_,v)| v.dims()[1] as usize).sum()).collect()
    }

    pub(crate) fn tensor(&self) -> MutexGuard<'_, Tensor> {
        self.tensor.lock().unwrap()
    }
    pub(crate) fn layers(&self) -> &Vec<Mutex<Layer>> {
        &self.layers
    }
    pub(crate) fn edges(&self) -> MutexGuard<'_, Edges> {
        self.edges.lock().unwrap()
    }
}