/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use std::sync::{Mutex, MutexGuard};
use crate::node::edges::Edges;
use crate::node::layer::Layer;
use crate::node::tensor::Tensor;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Mesh {
    layers: Vec<Mutex<Layer>>,
    tensor: Mutex<Tensor>,
    edges:  Mutex<Edges>,
}

impl Mesh {

    /// Mesh which contains Values, Layers and logical connection to Nodes.
    /// # Return Values
    /// Node Instance.
    pub(crate) fn new(layers: Vec<Mutex<Layer>>, edges: Mutex<Edges>) -> Self {
        Mesh { layers, tensor: Mutex::new(Default::default()), edges
        }
    }

    /// Returns the logical Topology of the Mesh.
    /// # Return Values
    /// Rows are represented as indices, Columns as the number in the Indices.
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