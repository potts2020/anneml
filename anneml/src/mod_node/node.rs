use std::sync::{Arc, Mutex };
use arrayfire::{Array, Dim4, dim4, join_many};
use uuid::Uuid;
use crate::mod_node::attribute::{Attribute, TensorDescriptor};
use crate::mod_node::edges::{Edges, NodeRange};
use crate::mod_node::layer::Layer;
use crate::mod_node::mesh::Mesh;
use crate::mod_node::mod_processor::processor::Processor;
use crate::mod_node::tensor::Tensor;
use crate::mod_node::utils::build_array;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Node{
    uuid: Uuid,
    mesh: Arc<Mesh>,
    children: Vec<Vec<Arc<Mutex<Node>>>>

}

impl Node {

    pub(crate) fn new(uuid: Uuid, node_type: NodeType) -> Self {
        let (layers,children, edges) = match node_type {
            NodeType::Leaf(input, hidden) => { Node::new_leaf_node(uuid, input, hidden) }
            NodeType::Vertex(input, child, output) => { Node::new_vertex_node(uuid, input, output, child) }
        };
        Node { uuid, mesh: Arc::new(Mesh::new(layers, edges)), children }
    }

        fn new_leaf_node(uuid: Uuid, input: Attribute, hidden: Vec<(Attribute, u64)>) -> (Vec<Mutex<Layer>>, Vec<Vec<Arc<Mutex<Node>>>>, Mutex<Edges>) {
            (Node::new_node_layers(uuid, input, hidden), vec![], Mutex::new(Edges::new(NodeRange::All)))
        }

        fn new_vertex_node(uuid: Uuid, input: Attribute, output: Attribute, child: Node) -> (Vec<Mutex<Layer>>, Vec<Vec<Arc<Mutex<Node>>>>, Mutex<Edges>) {
            (Node::new_node_layers(uuid, input, vec![(output, 0)]), vec![vec![Arc::new(Mutex::new(child))]], Mutex::new(Edges::new(NodeRange::Selective(vec![]))))
        }

        fn new_node_layers(uuid: Uuid, input: Attribute, hidden: Vec<(Attribute, u64)>) -> Vec<Mutex<Layer>> {
            let mut layers = vec![Mutex::new(Layer::new(input, Tensor::default()))];
            layers.splice(1..1, hidden.into_iter().map(|(attr, count)| {
                let mut hash_map = Tensor::default();
                if count > 0 { hash_map.insert(uuid.to_string().as_str(), Array::new_empty(Dim4::new(&[1, count, 1, 1]))); };
                Mutex::new(Layer::new(attr, hash_map))
            }).collect::<Vec<Mutex<Layer>>>());
            layers
        }

    pub(crate) fn add_child_to_parent(&mut self, child: Node, index: Option<usize>) -> Result<(), &'static str> {
        if self.is_leaf_node() { return Err("Unable to add child to leaf.") }
        let child = Arc::new(Mutex::new(child));
        match index {
            None => { self.children.push(vec![child]); }
            Some(index) => { self.children[index].push(child); }
        }
        Ok(())
    }

    pub(crate) fn index_into_node(&self, indices: &[(usize, usize)]) -> Result<Arc<Mutex<Node>>, &'static str> {
        if self.is_leaf_node() { return Err("Unable to index into leaf node."); }
        let node = self.children[indices[0].0][indices[0].1].clone();
        if indices[1..].is_empty() { Ok(node) } else { node.lock().unwrap().index_into_node(&indices[1..]) }
    }

    fn build_leaf(&self) {
        let topology = self.mesh().topology();
        self.mesh().tensor().insert("_SYSTEM_VALUES", build_array(&TensorDescriptor::Const(0.0), dim4!(1,*topology.iter().max().unwrap() as u64,topology.len() as u64,1)));
        self.mesh().layers().iter().enumerate().for_each(|(index, layer)| layer.lock().unwrap().build(index, &topology));
    }

    fn is_leaf_node(&self) -> bool {
        self.children.is_empty()
    }

    pub(crate) fn output(&self) -> Array<f64> {
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

pub(crate) enum NodeType {
    Leaf(Attribute, Vec<(Attribute, u64)>),
    Vertex(Attribute, Node, Attribute),
}