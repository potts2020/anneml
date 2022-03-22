/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::sync::{Arc, RwLock};
use crate::node::node::Node;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Network {
    node: Arc<RwLock<Node>>,
}

impl Network {
    /// Create a Network that contains Node(s).
    ///
    /// A Network is a logical container for a Node. Networks can save and load all information within itself.
    ///
    /// # Example(s)
    /// ```
    /// // Create a Attribute
    /// use uuid::Uuid;
    /// use anneml::node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
    /// use anneml::node::network::Network;
    /// use anneml::node::node::{Node, NodeType};
    /// use anneml::node::scope::Scope;
    ///
    /// let descriptor = vec![("WEIGHTS", TensorDescriptor::RandN),("BIASES", TensorDescriptor::Const(1.3))];
    /// let attribute = Attribute::new(Activation::Sigmoid, CellType::Mlp, descriptor, Scope::new(1,1));
    /// let leaf_node = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(), 3), (attribute.clone(), 2)]);
    /// let node = Node::new(Uuid::new_v4(), leaf_node);
    ///
    /// let network = Network::new(node);
    /// ```
    pub fn new(node: Node) -> Self {
        Network { node: Arc::new(RwLock::new(node)) }
    }

    /// Acquire a Networks wrapped node reference.
    pub fn node(&self) -> Arc<RwLock<Node>> {
        self.node.clone()
    }

    /// Saves a serialized .annml file.
    pub fn save(&self) {
        let f_name = format!("{}.annml", self.node().read().unwrap().uuid().to_hyphenated().to_string());
        let mut file = File::create( f_name).unwrap();
        let _result = file.write_all(&bincode::serialize(&self).unwrap());
    }

    /// Loads a serialized .annml file.
    pub fn load(f_name: &str) -> std::io::Result<Network> {
        let mut buffer : Vec<u8> = vec![];
        BufReader::new(File::open(f_name)?).read_to_end(&mut buffer)?;
        Ok(bincode::deserialize(&buffer).unwrap())
    }
}