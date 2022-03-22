/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use std::sync::{Arc, Mutex, MutexGuard};
use arrayfire::{add, Array, assign_seq, index, join_many, mul, seq, sigmoid, sum, tanh, transpose};
use crate::node::attribute::{Activation, CellType};
use crate::node::edges::NodeRange;
use crate::node::layer::Layer;
use crate::node::mesh::Mesh;
use crate::node::node::Node;
use crate::node::tensor::Tensor;

pub(crate) trait Processor {

    fn traverse(&self, tensor: &Tensor, build: bool);

    fn assign_children_inputs(&self, mut layers: &Vec<Mutex<Layer>>, child: &MutexGuard<Node>, build: bool) {
        child.traverse(&layers[0].lock().unwrap().tensor, build);
        layers[1].lock().unwrap().tensor.merge(&child.mesh().layers().last().unwrap().lock().unwrap().tensor, &child.mesh().edges().output_to_parent);
    }

    fn assign_sibling_inputs(&self, children: &Vec<Vec<Arc<Mutex<Node>>>>, child: &MutexGuard<Node>) {
        child.mesh().edges().input_from_peer_output.iter().for_each(|link|
            child.mesh().layers()[0].lock().unwrap().tensor.merge(&children[link.0.0][link.0.1].lock().unwrap().mesh().layers().last().unwrap().lock().unwrap().tensor, &link.1));
    }

    fn assign_parent_outputs(&self, mut tensor: MutexGuard<Tensor>, child: &MutexGuard<Node>) {
        child.mesh().tensor().hash_map.iter().filter(|(key,_)|
            match &child.mesh().edges().output_to_parent {
                NodeRange::Selective(key_chain) => { key_chain.contains(key) }
                _ => { true }
            }).for_each(|(key, value)| { if !key.contains("_SYSTEM") { tensor.insert(key, value.clone()); } }  );
    }

    fn process(&self, mesh: Arc<Mesh>, tensor: &Tensor) {

        // Filter the inputs by what is allowed in this layer.
        let filtered_inputs = join_many(1,
                                            tensor.hash_map.iter().filter(|(key, _)| {

                                                if key.contains("_SYSTEM") { return false }
                                                match &mesh.edges().input_from_parent {
                                                    NodeRange::Selective(key_chain) => { key_chain.contains(key) }
                                                    NodeRange::All => { true }
                                                }
                                            }).map(|(_, value)| value ).collect::<Vec<&Array<f64>>>()
        ); // Get the remaining Arrays.

        // Pass data in to update values inputs.

        assign_seq(mesh.tensor().hash_map.get_mut("_SYSTEM_VALUES").unwrap(), &[seq!(0,0,1), seq!(0,(filtered_inputs.dims()[1] - 1) as i32, 1), seq!(0, 0, 1)], &filtered_inputs);

        let topology = mesh.topology();

        for (i, layer) in mesh.layers().iter().enumerate() {
            let value_seq = [seq!(0,0,1), seq!(0,(topology[i] - 1) as i32, 1), seq!(i as i32, i as i32, 1)];
            let node_type_process: Array<f64> = node_type(layer.lock().unwrap().attribute.cell_type(), index(mesh.tensor().hash_map.get("_SYSTEM_VALUES").unwrap(), &value_seq));
            let activation_process = activation(layer.lock().unwrap().attribute.activation(), node_type_process);
            //Update Values
            assign_seq(mesh.tensor().hash_map.get_mut("_SYSTEM_VALUES").unwrap(), &value_seq, &activation_process);

            let data = layer.lock().unwrap().domain.domain_data();
            if i < mesh.layers().len() - 1 + data.0 {
                let seq = [seq!(0,0,1),seq!(0, (layer.lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims()[0] - 1) as i32,1),seq!(data.1 as i32,data.3 as i32,1)];
                let mul = mul(layer.lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap(), &activation_process, true);
                let sum = sum(&mul, 1);
                let plus = add(&sum, layer.lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap(), false);
                let transpose = transpose(&plus, false);
                let output = add(&index(mesh.tensor().hash_map.get("_SYSTEM_VALUES").unwrap(), &seq), &transpose,false);
                assign_seq(mesh.tensor().hash_map.get_mut("_SYSTEM_VALUES").unwrap(), &seq, &output);
            }
        }

        //Assign the output as a tensor entry, the key being the guid.
        // let d2 = (mesh.tensor().hash_map.get("_SYSTEM_VALUES").unwrap().dims()[2] - 1) as i32;
        // let seq = &[seq!(0,0,1), seq!(0,(mesh.topology().last().unwrap() - 1) as i32, 1), seq!(d2,d2,1)];
        // let indexed = index(mesh.tensor().hash_map.get("_SYSTEM_VALUES").unwrap(), seq);
        // mesh.tensor().hash_map.insert("_SYSTEM_VALUES".to_string(), indexed);
        // arrayfire::print(&mesh.tensor().hash_map.get("_SYSTEM_VALUES").unwrap());

    }
}

fn activation(activation: &Activation, array: Array<f64>) -> Array<f64>{
    match activation{
        Activation::Sigmoid  => { sigmoid(&array)}
        Activation::TanH     => { tanh(&array)}
        _ => { array }
    }
}

fn node_type(node_type: &CellType, array: Array<f64>) -> Array<f64>{
    match &node_type{
        CellType::Mlp => { array }
    }
}