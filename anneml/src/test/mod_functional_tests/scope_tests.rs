use arrayfire::{Array, dim4};
use uuid::Uuid;
use crate::mod_node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
use crate::mod_node::mod_processor::processor::Processor;
use crate::mod_node::network::Network;
use crate::mod_node::node::{Node, NodeType};
use crate::mod_node::scope::Scope;
use crate::mod_node::tensor::Tensor;

#[test]
fn create_layers_0_1(){
    let attributes = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attributes.clone(),
            vec![
                (attributes.clone(), 3),
                (attributes.clone(), 4),
                (attributes.clone(), 3),
                (attributes.clone(), 2),
                (attributes.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    assert_eq!((0,0,0,1,0),network.node().read().unwrap().mesh().layers()[0].lock().unwrap().domain.domain_data());
    assert_eq!((0,1,1,2,0),network.node().read().unwrap().mesh().layers()[1].lock().unwrap().domain.domain_data());
    assert_eq!((0,2,2,3,0),network.node().read().unwrap().mesh().layers()[2].lock().unwrap().domain.domain_data());
    assert_eq!((0,3,3,4,0),network.node().read().unwrap().mesh().layers()[3].lock().unwrap().domain.domain_data());
    assert_eq!((0,4,4,5,0),network.node().read().unwrap().mesh().layers()[4].lock().unwrap().domain.domain_data());
    assert_eq!((0,5,5,5,1),network.node().read().unwrap().mesh().layers()[5].lock().unwrap().domain.domain_data());
}

#[test]
fn create_layers_0_2(){
    let attributes = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,2));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attributes.clone(),
            vec![
                (attributes.clone(), 3),
                (attributes.clone(), 4),
                (attributes.clone(), 3),
                (attributes.clone(), 2),
                (attributes.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    assert_eq!((0,0,0,2,0),network.node().read().unwrap().mesh().layers()[0].lock().unwrap().domain.domain_data());
    assert_eq!((0,1,1,3,0),network.node().read().unwrap().mesh().layers()[1].lock().unwrap().domain.domain_data());
    assert_eq!((0,2,2,4,0),network.node().read().unwrap().mesh().layers()[2].lock().unwrap().domain.domain_data());
    assert_eq!((0,3,3,5,0),network.node().read().unwrap().mesh().layers()[3].lock().unwrap().domain.domain_data());
    assert_eq!((0,4,4,5,1),network.node().read().unwrap().mesh().layers()[4].lock().unwrap().domain.domain_data());
    assert_eq!((0,5,5,5,2),network.node().read().unwrap().mesh().layers()[5].lock().unwrap().domain.domain_data());
}

#[test]
fn create_layers_0_3(){
    let attributes = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,3));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attributes.clone(),
            vec![
                (attributes.clone(), 3),
                (attributes.clone(), 4),
                (attributes.clone(), 3),
                (attributes.clone(), 2),
                (attributes.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    assert_eq!((0,0,0,3,0),network.node().read().unwrap().mesh().layers()[0].lock().unwrap().domain.domain_data());
    assert_eq!((0,1,1,4,0),network.node().read().unwrap().mesh().layers()[1].lock().unwrap().domain.domain_data());
    assert_eq!((0,2,2,5,0),network.node().read().unwrap().mesh().layers()[2].lock().unwrap().domain.domain_data());
    assert_eq!((0,3,3,5,1),network.node().read().unwrap().mesh().layers()[3].lock().unwrap().domain.domain_data());
    assert_eq!((0,4,4,5,2),network.node().read().unwrap().mesh().layers()[4].lock().unwrap().domain.domain_data());
    assert_eq!((0,5,5,5,3),network.node().read().unwrap().mesh().layers()[5].lock().unwrap().domain.domain_data());
}

#[test]
fn create_layers_1_1(){
    let attributes = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(1,1));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attributes.clone(),
            vec![
                (attributes.clone(), 3),
                (attributes.clone(), 4),
                (attributes.clone(), 3),
                (attributes.clone(), 2),
                (attributes.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    assert_eq!((1,0,0,1,0),network.node().read().unwrap().mesh().layers()[0].lock().unwrap().domain.domain_data());
    assert_eq!((0,0,1,2,0),network.node().read().unwrap().mesh().layers()[1].lock().unwrap().domain.domain_data());
    assert_eq!((0,1,2,3,0),network.node().read().unwrap().mesh().layers()[2].lock().unwrap().domain.domain_data());
    assert_eq!((0,2,3,4,0),network.node().read().unwrap().mesh().layers()[3].lock().unwrap().domain.domain_data());
    assert_eq!((0,3,4,5,0),network.node().read().unwrap().mesh().layers()[4].lock().unwrap().domain.domain_data());
    assert_eq!((0,4,5,5,1),network.node().read().unwrap().mesh().layers()[5].lock().unwrap().domain.domain_data());
}

#[test]
fn create_layers_2_2(){
    let attributes = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(2,2));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attributes.clone(),
            vec![
                (attributes.clone(), 3),
                (attributes.clone(), 4),
                (attributes.clone(), 3),
                (attributes.clone(), 2),
                (attributes.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    assert_eq!((2,0,0,2,0),network.node().read().unwrap().mesh().layers()[0].lock().unwrap().domain.domain_data());
    assert_eq!((1,0,1,3,0),network.node().read().unwrap().mesh().layers()[1].lock().unwrap().domain.domain_data());
    assert_eq!((0,0,2,4,0),network.node().read().unwrap().mesh().layers()[2].lock().unwrap().domain.domain_data());
    assert_eq!((0,1,3,5,0),network.node().read().unwrap().mesh().layers()[3].lock().unwrap().domain.domain_data());
    assert_eq!((0,2,4,5,1),network.node().read().unwrap().mesh().layers()[4].lock().unwrap().domain.domain_data());
    assert_eq!((0,3,5,5,2),network.node().read().unwrap().mesh().layers()[5].lock().unwrap().domain.domain_data());
}

#[test]
fn create_layers_3_3(){
    let attributes = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(3,3));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attributes.clone(),
            vec![
                (attributes.clone(), 3),
                (attributes.clone(), 4),
                (attributes.clone(), 3),
                (attributes.clone(), 2),
                (attributes.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    assert_eq!((3,0,0,3,0),network.node().read().unwrap().mesh().layers()[0].lock().unwrap().domain.domain_data());
    assert_eq!((2,0,1,4,0),network.node().read().unwrap().mesh().layers()[1].lock().unwrap().domain.domain_data());
    assert_eq!((1,0,2,5,0),network.node().read().unwrap().mesh().layers()[2].lock().unwrap().domain.domain_data());
    assert_eq!((0,0,3,5,1),network.node().read().unwrap().mesh().layers()[3].lock().unwrap().domain.domain_data());
    assert_eq!((0,1,4,5,2),network.node().read().unwrap().mesh().layers()[4].lock().unwrap().domain.domain_data());
    assert_eq!((0,2,5,5,3),network.node().read().unwrap().mesh().layers()[5].lock().unwrap().domain.domain_data());
}