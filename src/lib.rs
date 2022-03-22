//! Anneml is a composite machine learning library designed for simplicity and ease of use.
//!
//! Most other machine learning libraries focus on building a single monolithic network. Anneml joins many networks together that get trained independently.
//!
//! These are the Anneml advantages:
//! - Write once and run anywhere architecture.
//! - Smaller Networks can be trained to complete a single task and joined together for more complex behavior.
//! - Modifying one aspect of the system is time and cost-efficient.
//! - Networks are improved independently from the whole system.
//! - Troubleshooting functionality is less time-intensive.
//!
//!
//!
//! # My first Neural Network
//! ```
//! use uuid::Uuid;
//! use anneml::node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
//! use anneml::node::network::Network;
//! use anneml::node::node::{Node, NodeType};
//! use anneml::node::scope::Scope;
//!
//! let descriptor = vec![("WEIGHTS", TensorDescriptor::RandN),("BIASES", TensorDescriptor::Const(1.3))];
//! let attribute = Attribute::new(Activation::Sigmoid, CellType::Mlp, descriptor, Scope::new(1,1));
//! let leaf_node = NodeType::Leaf(attribute.clone(), vec![(attribute.clone(), 3), (attribute.clone(), 2)]);
//! let node = Node::new(Uuid::new_v4(), leaf_node);
//!
//! let network = Network::new(node);
//! ```
#[cfg(test)]
mod test {
    mod functional_tests;
}

pub mod node;

