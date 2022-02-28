use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::sync::{Arc, RwLock};
use crate::mod_node::node::Node;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Network {
    node: Arc<RwLock<Node>>,


}

impl Network {
    pub(crate) fn new(node: Node) -> Self {
        Network { node: Arc::new(RwLock::new(node)) }
    }

    pub(crate) fn node(&self) -> Arc<RwLock<Node>> {
        self.node.clone()
    }

    /// Saves a tensor file in a .ann serialized format.
    /// # Return Values
    /// OK().
    pub(crate) fn save(&self) {
        let f_name = format!("{}.annml", self.node().read().unwrap().uuid().to_hyphenated().to_string());
        let mut file = File::create( f_name).unwrap();
        let _result = file.write_all(&bincode::serialize(&self).unwrap());
    }

    /// Loads a tensor file.
    /// # Return Values
    /// OK(Tensor).
    pub(crate) fn load(f_name: &str) -> std::io::Result<Network> {
        let mut buffer : Vec<u8> = vec![];
        BufReader::new(File::open(f_name)?).read_to_end(&mut buffer)?;
        Ok(bincode::deserialize(&buffer).unwrap())
    }
}