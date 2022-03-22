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
    /// A Node is a wrapper for a Mesh.
    ///
    /// The Node can either be a leaf or vertex, which will determine the role of its child Mesh.
    /// A Node with no children is a leaf- and vertex if it has children.
    /// Nodes connect between themselves, which form the structure of a Network.
    ///
    /// # Example(s)
    ///
    /// ```
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let leaf_node = Node::new(Uuid::new_v4(), node_type);
    /// assert_eq!(true, leaf_node.is_leaf_node());
    ///
    /// // Create a Vertex Node
    /// let node_type = NodeType::Vertex(attribute.clone(),leaf_node, attribute.clone());
    /// let vertex_node = Node::new(Uuid::new_v4(), node_type);
    /// assert_eq!(false, vertex_node.is_leaf_node());
    /// ```
    pub fn new(uuid: Uuid, node_type: NodeType) -> Self {
        let (layers,children, edges) = match node_type {
            NodeType::Leaf(input, hidden) => { Node::derive_leaf_components(uuid, input, hidden) }
            NodeType::Vertex(input, child, output) => { Node::derive_vertex_components(uuid, input, output, child) }
        };
        Node { uuid, mesh: Arc::new(Mesh::new(layers, edges)), children }
    }

        /// Returns components required for creating a leaf node.
        ///
        /// A Node is a leaf if it does not (and can not) have any child. Returned components that are required to build a leaf node are the following:
        /// 1) Layers which are used to define the size of a Node's Mesh.
        /// 2) Empty child array that is the determinant of a Leaf Node.
        /// 3) Edges which connect the Node together.
        fn derive_leaf_components(uuid: Uuid, input: Attribute, hidden: Vec<(Attribute, u64)>) -> (Vec<Mutex<Layer>>, Vec<Vec<Arc<Mutex<Node>>>>, Mutex<Edges>) {
            (Node::new_node_layers(uuid, input, hidden), vec![], Mutex::new(Edges::new(NodeRange::All)))
        }

        /// Returns components required for creating a vertex node.
        ///
        /// A Node is a leaf if it does not (and can not) have any child. Returned components that are required to build a leaf node are the following:
        /// 1) Layers which are used to define the size of a Node's Mesh.
        /// 2) Populated child array that is the determinant of a Vertex Node.
        /// 3) Edges which connect the Node together.
        fn derive_vertex_components(uuid: Uuid, input: Attribute, output: Attribute, child: Node) -> (Vec<Mutex<Layer>>, Vec<Vec<Arc<Mutex<Node>>>>, Mutex<Edges>) {
            (Node::new_node_layers(uuid, input, vec![(output, 0)]), vec![vec![Arc::new(Mutex::new(child))]], Mutex::new(Edges::new(NodeRange::Selective(vec![]))))
        }

        /// Builds relational layers.
        ///
        /// Creates a vector of layers that may interact with each other through edges.
        /// The first layer is an input layer that is assigned an attribute and a default Tensor.
        /// All other layers are assigned a Tensor with a Arrayfire array entry that is of size (1, count, 1,1), which is the layer size.
        /// We do not assign a Tensor entry to the input layer because the dimensions of the Arrayfire array will change as edges are set.
        fn new_node_layers(uuid: Uuid, attribute: Attribute, hidden: Vec<(Attribute, u64)>) -> Vec<Mutex<Layer>> {
            let mut layers = vec![Mutex::new(Layer::new(attribute, Tensor::default()))];
            layers.splice(1..1, hidden.into_iter().map(|(attr, count)| {
                let mut hash_map = Tensor::default();
                if count > 0 { hash_map.insert(uuid.to_string().as_str(), Array::new_empty(Dim4::new(&[1, count, 1, 1]))); };
                Mutex::new(Layer::new(attr, hash_map))
            }).collect::<Vec<Mutex<Layer>>>());
            layers
        }


    /// Assign a child Node to a parent Vertex node.
    ///
    /// The passed in child Node is consumed and assigned to the parent.
    /// The index Option determines if the node will be appended to an existing column of children (Some) or a New Column (None)
    ///
    /// # Errors
    /// Depending on whether the Node is a Leaf or a Vertex, we will return one of two results:
    /// 1) If the Node is a Vertex:
    ///     - And index is Some(`index`) where `index` is within bounds of the current number of columns in the child vector, we will add a child to that column and return ok(())
    ///     - And index is None, we append a new column with the child inside.
    /// 2) If the Node is a Leaf: We receive an Err of "_Unable to add child to leaf._".
    /// # Panics
    /// Panics if `index` is out of bounds.
    ///
    /// # Example(s)
    /// ```
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Add a new column
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let leaf_node = Node::new(Uuid::new_v4(), node_type);
    /// // Create a Vertex Node, assign existing Leaf to Vertex Node
    /// let node_type = NodeType::Vertex(attribute.clone(),leaf_node, attribute.clone());
    /// let mut vertex_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Create a NEW Leaf Node
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let new_leaf_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Assign new leaf Node to new vertex_node child column
    /// assert_eq!(1, vertex_node.children().len());
    /// let result = vertex_node.add_child_to_parent(new_leaf_node, None);
    /// assert_eq!(Ok(()), result);
    /// assert_eq!(2, vertex_node.children().len());
    /// ```
    /// ---
    /// ```
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Append to an existing column
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let leaf_node = Node::new(Uuid::new_v4(), node_type);
    /// // Create a Vertex Node, assign existing Leaf to Vertex Node
    /// let node_type = NodeType::Vertex(attribute.clone(),leaf_node, attribute.clone());
    /// let mut vertex_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Create a NEW Leaf Node
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let new_leaf_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Assign new leaf Node to new vertex_node child column
    /// assert_eq!(1, vertex_node.children()[0].len());
    /// let result = vertex_node.add_child_to_parent(new_leaf_node, Some(0));
    /// assert_eq!(Ok(()), result);
    /// assert_eq!(2, vertex_node.children()[0].len());
    /// ```
    /// ---
    /// ```
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Append add to leaf node
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let mut leaf_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Create a NEW Leaf Node
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let new_leaf_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Assign new leaf Node to new vertex_node child column
    /// assert_eq!(0, leaf_node.children().len());
    /// let result = leaf_node.add_child_to_parent(new_leaf_node, Some(0));
    /// assert_eq!(Err("Unable to add child to leaf."), result);
    /// assert_eq!(0, leaf_node.children().len());
    /// ```
    /// ---
    /// ```should_panic
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Panic
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let leaf_node = Node::new(Uuid::new_v4(), node_type);
    /// // Create a Vertex Node, assign existing Leaf to Vertex Node
    /// let node_type = NodeType::Vertex(attribute.clone(),leaf_node, attribute.clone());
    /// let mut vertex_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Create a NEW Leaf Node
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let new_leaf_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Assign new leaf Node to new vertex_node child column
    /// let result = vertex_node.add_child_to_parent(new_leaf_node, Some(1));
    /// assert!(false);
    /// ```
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
    /// Indices are the (X,Y) position of the Node to index into and that nodes subsequent children.
    ///
    /// # Errors
    /// Depending on whether the Node is a Leaf or a Vertex, we will return one of two results:
    /// 1) If the Node is a Vertex:
    ///     - And indices: &[(usize, usize)] _do not_ exceed the depth of the network, a Result<Arc<Mutex<Node>> will be returned.
    ///     - And indices: &[(usize, usize)] _does_ exceed the depth of the network, we panic.
    /// 2) If the Node is a Leaf: We receive an Err of "_Unable to index into leaf node._".
    /// # Panics
    /// Panics if recursive `index` is out of bounds.
    ///
    /// # Example(s)
    /// ```
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Index into a Vertex Nodes' child
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let leaf_node = Node::new(Uuid::new_v4(), node_type);
    /// // Create a Vertex Node, assign existing Leaf to Vertex Node
    /// let node_type = NodeType::Vertex(attribute.clone(),leaf_node, attribute.clone());
    /// let mut vertex_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Get Leaf Node &reference
    /// let leaf_node = vertex_node.index_into_node(&[(0,0)]);
    /// assert_eq!(true, leaf_node.unwrap().lock().unwrap().is_leaf_node());
    /// ```
    /// ---
    /// ```
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Index into a Leaf Node
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let leaf_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Assign new leaf Node to new vertex_node child column
    /// let result = leaf_node.index_into_node(&[(0,0)]);
    /// assert!(result.is_err());
    /// ```
    /// ---
    /// ```should_panic
    /// use uuid::Uuid;
    /// use anneml::node::attribute::Attribute;
    /// use anneml::node::node::{Node, NodeType};
    /// // Index into a Vertex Nodes' child
    /// // Create a Leaf Node
    /// let attribute = Attribute::default();
    /// let node_type = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(),2),(attribute.clone(),3)]);
    /// let leaf_node = Node::new(Uuid::new_v4(), node_type);
    /// // Create a Vertex Node, assign existing Leaf to Vertex Node
    /// let node_type = NodeType::Vertex(attribute.clone(),leaf_node, attribute.clone());
    /// let mut vertex_node = Node::new(Uuid::new_v4(), node_type);
    ///
    /// // Get Leaf Node &reference (should panic)
    /// let leaf_node = vertex_node.index_into_node(&[(0,0),(0,0)]);
    /// assert!(false);
    /// ```
    pub fn index_into_node(&self, indices: &[(usize, usize)]) -> Result<Arc<Mutex<Node>>, &'static str> {
        if self.is_leaf_node() { return Err("Unable to index into leaf node."); }
        let node = self.children[indices[0].0][indices[0].1].clone();
        if indices[1..].is_empty() { Ok(node) } else { node.lock().unwrap().index_into_node(&indices[1..]) }
    }
    
    /// Instantiates values for a leaf node.
    ///
    /// Create a _SYSTEM_VALUES entry in the Nodes Tensor. _SYSTEM_VALUES tracks the values of each layer.
    /// Build layers within Nodes Mesh.
    fn init_leaf(&self) {
        let topology = self.mesh().topology();
        self.mesh().tensor().insert("_SYSTEM_VALUES", build_array(&TensorDescriptor::Const(0.0), dim4!(1,*topology.iter().max().unwrap() as u64,topology.len() as u64,1)));
        self.mesh().layers().iter().enumerate().for_each(|(index, layer)| layer.lock().unwrap().build(index, &topology));
    }

    /// Returns whether a Node is a Leaf.
    ///
    /// A Node is a leaf if it does not have any children.
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

    /// Acquire reference to Node Children Vector.
    pub fn children(&self) -> &Vec<Vec<Arc<Mutex<Node>>>> {
        &self.children
    }
}

impl Processor for Node {
    /// Recursively traverse through child nodes and instantiate values..
    /// tensors passed into traverse are assigned to the input layer of the Node.
    fn traverse(&self, tensor: &Tensor, build: bool) {
        self.mesh().layers()[0].lock().unwrap().tensor.merge(tensor, &self.mesh().edges().input_from_parent);
        match self.is_leaf_node() {
            true =>  { if build { self.init_leaf() } self.process(self.mesh(), tensor) }
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