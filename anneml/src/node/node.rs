/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use std::sync::{Arc, Mutex };
use arrayfire::{Array, Dim4, dim4, join_many};
use uuid::Uuid;
use crate::node::attribute::{Attribute, TensorDescriptor};
use crate::node::edges::{Edges, NodeRange};
use crate::node::layer::Layer;
use crate::node::mesh::Mesh;
use crate::node::processor::processor::Processor;
use crate::node::tensor::Tensor;
use crate::node::utils::build_array;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Node{
    uuid: Uuid,
    mesh: Arc<Mesh>,
    children: Vec<Vec<Arc<Mutex<Node>>>>

}

impl Node {
    /// Nodes are wrappers for either a Mesh.
    ///
    /// Nodes are responsible for interpreting the role their child mesh has in the larger Neural Network.
    /// Nodes are either a Leaf or Vertex, Vertex nodes have children while Leaves do not.
    /// Nodes are able to make logical connections between Sibling, Children and Parent Nodes that allow many Neural networks to interface with each other.
    ///
    /// # Return Values
    /// Node
    pub fn new(uuid: Uuid, node_type: NodeType) -> Self {
        let (layers,children, edges) = match node_type {
            NodeType::Leaf(input, hidden) => { Node::new_leaf_node(uuid, input, hidden) }
            NodeType::Vertex(input, child, output) => { Node::new_vertex_node(uuid, input, output, child) }
        };
        Node { uuid, mesh: Arc::new(Mesh::new(layers, edges)), children }
    }

        /// Builds components for a Leaf Node.
        ///
        /// Leaf nodes are Nodes that do not have Children.
        ///
        /// # Return Values
        /// Layers used in the Construction of the Nodes Mesh.
        /// Empty child array that differentiates Leaf from Vertex Nodes.
        /// Edges that logically connect the Node to its Parent, or Siblings.
        fn new_leaf_node(uuid: Uuid, input: Attribute, hidden: Vec<(Attribute, u64)>) -> (Vec<Mutex<Layer>>, Vec<Vec<Arc<Mutex<Node>>>>, Mutex<Edges>) {
            (Node::new_node_layers(uuid, input, hidden), vec![], Mutex::new(Edges::new(NodeRange::All)))
        }

        /// Builds components for a Vertex Node.
        ///
        /// Leaf nodes are Nodes that have Children.
        ///
        /// # Return Values
        /// Layers used in the Construction of the Nodes Mesh.
        /// Child array that differentiates Leaf from Vertex Nodes.
        /// Edges that logically connect the Node to its Parent, or Siblings.
        fn new_vertex_node(uuid: Uuid, input: Attribute, output: Attribute, child: Node) -> (Vec<Mutex<Layer>>, Vec<Vec<Arc<Mutex<Node>>>>, Mutex<Edges>) {
            (Node::new_node_layers(uuid, input, vec![(output, 0)]), vec![vec![Arc::new(Mutex::new(child))]], Mutex::new(Edges::new(NodeRange::Selective(vec![]))))
        }

        /// Builds Layers that Map the relationship between a Neural Networks Inputs and Outputs.
        ///
        /// # Return Values
        /// Vector of Layers used in the creation of a Nodes Mesh.
        fn new_node_layers(uuid: Uuid, input: Attribute, hidden: Vec<(Attribute, u64)>) -> Vec<Mutex<Layer>> {
            let mut layers = vec![Mutex::new(Layer::new(input, Tensor::default()))];
            layers.splice(1..1, hidden.into_iter().map(|(attr, count)| {
                let mut hash_map = Tensor::default();
                if count > 0 { hash_map.insert(uuid.to_string().as_str(), Array::new_empty(Dim4::new(&[1, count, 1, 1]))); };
                Mutex::new(Layer::new(attr, hash_map))
            }).collect::<Vec<Mutex<Layer>>>());
            layers
        }

    /// Assigns a Node to a Parent Vertex Node.
    ///
    /// The passed in child Node is consumed and assigned to the parent.
    /// The index Option determines if the node will be appended to an existing column of children (Some) or a New Column (None)
    ///
    /// # Return Values
    /// Result which will fail or pass validation.
    pub fn add_child_to_parent(&mut self, child: Node, index: Option<usize>) -> Result<(), &'static str> {
        if self.is_leaf_node() { return Err("Unable to add child to leaf.") }
        let child = Arc::new(Mutex::new(child));
        match index {
            None => { self.children.push(vec![child]); }
            Some(index) => { self.children[index].push(child); }
        }
        Ok(())
    }

    /// Recursively Indexes into Nodes.
    ///
    /// Indices are the (X,Y) position of the Node to index into and that nodes subsequent child position.
    ///
    /// # Return Values
    /// A reference of a nested Node that can be modified.
    pub fn index_into_node(&self, indices: &[(usize, usize)]) -> Result<Arc<Mutex<Node>>, &'static str> {
        if self.is_leaf_node() { return Err("Unable to index into leaf node."); }
        let node = self.children[indices[0].0][indices[0].1].clone();
        if indices[1..].is_empty() { Ok(node) } else { node.lock().unwrap().index_into_node(&indices[1..]) }
    }

    /// Commit and instantiate the values for Leaf Nodes on traversal.
    fn build_leaf(&self) {
        let topology = self.mesh().topology();
        self.mesh().tensor().insert("_SYSTEM_VALUES", build_array(&TensorDescriptor::Const(0.0), dim4!(1,*topology.iter().max().unwrap() as u64,topology.len() as u64,1)));
        self.mesh().layers().iter().enumerate().for_each(|(index, layer)| layer.lock().unwrap().build(index, &topology));
    }

    /// Returns whether a Node is a Leaf.
    pub fn is_leaf_node(&self) -> bool {
        self.children.is_empty()
    }

    /// Grabs the output values from children nodes and join them in a single column.
    pub fn output(&self) -> Array<f64> {
        join_many(1, self.mesh().tensor().hash_map.iter().filter(|(k, v)| !k.contains("_SYSTEM")).map(|(k, v)| v).collect::<Vec<&Array<f64>>>())
    }

    pub(crate) fn mesh(&self) -> Arc<Mesh> {
        self.mesh.clone()
    }

    pub(crate) fn uuid(&self) -> Uuid {
        self.uuid
    }
}

impl Processor for Node {
    /// Recursively traverse through child nodes and instantiate values..
    /// tensors passed into traverse are assigned to the input layer of the Node.
    fn traverse(&self, tensor: &Tensor, build: bool) {
        self.mesh().layers()[0].lock().unwrap().tensor.merge(tensor, &self.mesh().edges().input_from_parent);

        match self.is_leaf_node() {
            true =>  { if build { self.build_leaf() } self.process(self.mesh(), tensor) }
            false => {
                let children = &self.children.iter().flatten().map(|child| child.lock().unwrap());
                children.clone().for_each(|child| { self.assign_children_inputs(self.mesh().layers(), &child, build); });
                children.clone().for_each(|child| { self.assign_sibling_inputs(&self.children, &child); });
                children.clone().for_each(|child| { self.assign_parent_outputs(self.mesh().tensor(), &child); });
            }
        }
    }
}

/// Defines the values associated with Leaf and Vertex Nodes.
pub enum NodeType {
    Leaf(Attribute, Vec<(Attribute, u64)>),
    Vertex(Attribute, Node, Attribute),
}