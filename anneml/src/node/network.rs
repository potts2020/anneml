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
    /// Return an instance of a Network.
    ///
    /// Networks are the top-level components in the hierarchy of Neural Network components.
    /// A Network is assigned a Node as it's child which contains the logical descriptors of the Neural Network.
    ///
    /// # Return Values
    /// Network
    pub fn new(node: Node) -> Self {
        Network { node: Arc::new(RwLock::new(node)) }
    }

    /// Acquire a Network node reference.
    /// # Return Values
    /// Arc<Rwlock<Node>>
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
    /// # Return Values
    /// Result<Network>
    pub fn load(f_name: &str) -> std::io::Result<Network> {
        let mut buffer : Vec<u8> = vec![];
        BufReader::new(File::open(f_name)?).read_to_end(&mut buffer)?;
        Ok(bincode::deserialize(&buffer).unwrap())
    }
}