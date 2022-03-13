/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use arrayfire::{Array, assign_seq, constant, index, seq, dim4};

use std::ops::Mul;
use uuid::Uuid;
use crate::node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
use crate::node::processor::processor::Processor;
use crate::node::network::Network;
use crate::node::node::{Node, NodeType};
use crate::node::scope::Scope;
use crate::node::tensor::Tensor;

#[test]
fn create_weights_0_1(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 3),
                (attribute.clone(), 4),
                (attribute.clone(), 5),
                (attribute.clone(), 3)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    let w_dims = dim4!(5,5,2,5);
    let test_weights_cutouts : Array<f64> = constant(1.0, w_dims);

    let mut w_test_0 : Array<f64> = constant(0.0, dim4!(3,2,2,1));
    let seq_0_1 = [seq!(0,2,1),seq!(0,1,1),seq!(1,1,1)];
    assign_seq(&mut w_test_0, &seq_0_1, &index(&test_weights_cutouts, &seq_0_1));
    let expected = w_test_0.mul(network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_1 : Array<f64> = constant(0.0, dim4!(4,3,2,1));
    let seq_1_1 = [seq!(0,3,1),seq!(0,2,1),seq!(1,1,1)];
    assign_seq(&mut w_test_1, &seq_1_1, &index(&test_weights_cutouts,&seq_1_1));
    let expected = w_test_1.mul(network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_2 : Array<f64> = constant(0.0, dim4!(5,4,2,1));
    let seq_2_1 = [seq!(0,4,1),seq!(0,3,1),seq!(1,1,1)];
    assign_seq(&mut w_test_2, &seq_2_1, &index(&test_weights_cutouts,&seq_2_1));
    let expected = w_test_2.mul(network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_3 : Array<f64> = constant(0.0, dim4!(3,5,2,1));
    let seq_3_1 = [seq!(0,2,1),seq!(0,4,1),seq!(1,1,1)];
    assign_seq(&mut w_test_3, &seq_3_1, &index(&test_weights_cutouts,&seq_3_1));
    let expected = w_test_3.mul(network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());
}

#[test]
fn create_weights_0_2(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,2));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 3),
                (attribute.clone(), 4),
                (attribute.clone(), 5),
                (attribute.clone(), 3)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let w_dims = dim4!(5,5,3,5);
    let test_weights_cutouts : Array<f64> = constant(1.0, w_dims);

    let mut w_test_0 : Array<f64> = constant(0.0, dim4!(4,2,3,1));
    let seq_0_1 = [seq!(0,2,1),seq!(0,1,1),seq!(1,1,1)];
    assign_seq(&mut w_test_0, &seq_0_1, &index(&test_weights_cutouts, &seq_0_1));
    let seq_0_2 = [seq!(0,3,1),seq!(0,1,1),seq!(2,2,1)];
    assign_seq(&mut w_test_0, &seq_0_2, &index(&test_weights_cutouts, &seq_0_2));
    let expected = w_test_0.mul(network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_1 : Array<f64> = constant(0.0, dim4!(5,3,3,1));
    let seq_1_1 = [seq!(0,3,1),seq!(0,2,1),seq!(1,1,1)];
    assign_seq(&mut w_test_1, &seq_1_1, &index(&test_weights_cutouts,&seq_1_1));
    let seq_1_2 = [seq!(0,4,1),seq!(0,2,1),seq!(2,2,1)];
    assign_seq(&mut w_test_1, &seq_1_2, &index(&test_weights_cutouts,&seq_1_2));
    let expected = w_test_1.mul(network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_2 : Array<f64> = constant(0.0, dim4!(5,4,3,1));
    let seq_2_1 = [seq!(0,4,1),seq!(0,3,1),seq!(1,1,1)];
    assign_seq(&mut w_test_2, &seq_2_1, &index(&test_weights_cutouts,&seq_2_1));
    let seq_2_1 = [seq!(0,2,1),seq!(0,3,1),seq!(2,2,1)];
    assign_seq(&mut w_test_2, &seq_2_1, &index(&test_weights_cutouts,&seq_2_1));
    let expected = w_test_2.mul(network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_3 : Array<f64> = constant(0.0, dim4!(3,5,2,1));
    let seq_3_1 = [seq!(0,2,1),seq!(0,4,1),seq!(1,1,1)];
    assign_seq(&mut w_test_3, &seq_3_1, &index(&test_weights_cutouts,&seq_3_1));
    let expected = w_test_3.mul(network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());
}

#[test]
fn create_weights_0_3(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,3));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 3),
                (attribute.clone(), 4),
                (attribute.clone(), 5),
                (attribute.clone(), 3)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let w_dims = dim4!(5,5,4,5);
    let test_weights_cutouts : Array<f64> = constant(1.0, w_dims);

    let mut w_test_0 : Array<f64> = constant(0.0, dim4!(5,2,4,1));
    let seq_0_1 = [seq!(0,2,1),seq!(0,1,1),seq!(1,1,1)];
    assign_seq(&mut w_test_0, &seq_0_1, &index(&test_weights_cutouts, &seq_0_1));
    let seq_0_2 = [seq!(0,3,1),seq!(0,1,1),seq!(2,2,1)];
    assign_seq(&mut w_test_0, &seq_0_2, &index(&test_weights_cutouts, &seq_0_2));
    let seq_0_3 = [seq!(0,4,1),seq!(0,1,1),seq!(3,3,1)];
    assign_seq(&mut w_test_0, &seq_0_3, &index(&test_weights_cutouts, &seq_0_3));
    let expected = w_test_0.mul(network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_1 : Array<f64> = constant(0.0, dim4!(5,3,4,1));
    let seq_1_1 = [seq!(0,3,1),seq!(0,2,1),seq!(1,1,1)];
    assign_seq(&mut w_test_1, &seq_1_1, &index(&test_weights_cutouts,&seq_1_1));
    let seq_1_2 = [seq!(0,4,1),seq!(0,2,1),seq!(2,2,1)];
    assign_seq(&mut w_test_1, &seq_1_2, &index(&test_weights_cutouts,&seq_1_2));
    let seq_1_3 = [seq!(0,2,1),seq!(0,2,1),seq!(3,3,1)];
    assign_seq(&mut w_test_1, &seq_1_3, &index(&test_weights_cutouts,&seq_1_3));
    let expected = w_test_1.mul(network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_2 : Array<f64> = constant(0.0, dim4!(5,4,3,1));
    let seq_2_1 = [seq!(0,4,1),seq!(0,3,1),seq!(1,1,1)];
    assign_seq(&mut w_test_2, &seq_2_1, &index(&test_weights_cutouts,&seq_2_1));
    let seq_2_1 = [seq!(0,2,1),seq!(0,3,1),seq!(2,2,1)];
    assign_seq(&mut w_test_2, &seq_2_1, &index(&test_weights_cutouts,&seq_2_1));
    let expected = w_test_2.mul(network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_3 : Array<f64> = constant(0.0, dim4!(3,5,2,1));
    let seq_3_1 = [seq!(0,2,1),seq!(0,4,1),seq!(1,1,1)];
    assign_seq(&mut w_test_3, &seq_3_1, &index(&test_weights_cutouts,&seq_3_1));
    let expected = w_test_3.mul(network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());
}

#[test]
fn create_weights_1_1(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(1,1));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 3),
                (attribute.clone(), 4),
                (attribute.clone(), 5),
                (attribute.clone(), 3)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let w_dims = dim4!(5,5,3,5);
    let test_weights_cutouts : Array<f64> = constant(1.0, w_dims);

    let mut w_test_0 : Array<f64> = constant(0.0, dim4!(3,2,2,1));
    let seq_0_2 = [seq!(0,2,1),seq!(0,1,1),seq!(1,1,1)];
    assign_seq(&mut w_test_0, &seq_0_2, &index(&test_weights_cutouts, &seq_0_2));
    let expected = w_test_0.mul(network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_1 : Array<f64> = constant(0.0, dim4!(4,3,3,1));
    let seq_1_0 = [seq!(0,1,1),seq!(0,2,1),seq!(0,0,1)];
    assign_seq(&mut w_test_1, &seq_1_0, &index(&test_weights_cutouts,&seq_1_0));
    let seq_1_2 = [seq!(0,3,1),seq!(0,2,1),seq!(2,2,1)];
    assign_seq(&mut w_test_1, &seq_1_2, &index(&test_weights_cutouts,&seq_1_2));
    let expected = w_test_1.mul(network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_2 : Array<f64> = constant(0.0, dim4!(5,4,3,1));
    let seq_2_0 = [seq!(0,2,1),seq!(0,3,1),seq!(0,0,1)];
    assign_seq(&mut w_test_2, &seq_2_0, &index(&test_weights_cutouts,&seq_2_0));
    let seq_2_2 = [seq!(0,4,1),seq!(0,3,1),seq!(2,2,1)];
    assign_seq(&mut w_test_2, &seq_2_2, &index(&test_weights_cutouts,&seq_2_2));
    let expected = w_test_2.mul(network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_3 : Array<f64> = constant(0.0, dim4!(4,5,3,1));
    let seq_3_0 = [seq!(0,3,1),seq!(0,4,1),seq!(0,0,1)];
    assign_seq(&mut w_test_3, &seq_3_0, &index(&test_weights_cutouts,&seq_3_0));
    let seq_3_2 = [seq!(0,2,1),seq!(0,4,1),seq!(2,2,1)];
    assign_seq(&mut w_test_3, &seq_3_2, &index(&test_weights_cutouts,&seq_3_2));
    let expected = w_test_3.mul(network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_4 : Array<f64> = constant(0.0, dim4!(5,3,2,1));
    let seq_4_0 = [seq!(0,4,1),seq!(0,2,1),seq!(0,0,1)];
    assign_seq(&mut w_test_4, &seq_4_0, &index(&test_weights_cutouts,&seq_4_0));
    let expected = w_test_4.mul(network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());
}

#[test]
fn create_weights_2_2(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(2,2));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 3),
                (attribute.clone(), 4),
                (attribute.clone(), 5),
                (attribute.clone(), 3)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let w_dims = dim4!(5,5,5,5);
    let test_weights_cutouts : Array<f64> = constant(1.0, w_dims);

    let mut w_test_0 : Array<f64> = constant(0.0, dim4!(4,2,3,1));
    let seq_0_3 = [seq!(0,2,1),seq!(0,1,1),seq!(1,1,1)];
    assign_seq(&mut w_test_0, &seq_0_3, &index(&test_weights_cutouts, &seq_0_3));
    let seq_0_4 = [seq!(0,3,1),seq!(0,1,1),seq!(2,2,1)];
    assign_seq(&mut w_test_0, &seq_0_4, &index(&test_weights_cutouts, &seq_0_4));
    let expected = w_test_0.mul(network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_1 : Array<f64> = constant(0.0, dim4!(5,3,4,1));
    let seq_1_1 = [seq!(0,1,1),seq!(0,2,1),seq!(0,0,1)];
    assign_seq(&mut w_test_1, &seq_1_1, &index(&test_weights_cutouts,&seq_1_1));
    let seq_1_3 = [seq!(0,3,1),seq!(0,2,1),seq!(2,2,1)];
    assign_seq(&mut w_test_1, &seq_1_3, &index(&test_weights_cutouts,&seq_1_3));
    let seq_1_4 = [seq!(0,4,1),seq!(0,2,1),seq!(3,3,1)];
    assign_seq(&mut w_test_1, &seq_1_4, &index(&test_weights_cutouts,&seq_1_4));
    let expected = w_test_1.mul(network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_2 : Array<f64> = constant(0.0, dim4!(5,4,5,1));
    let seq_2_0 = [seq!(0,1,1),seq!(0,3,1),seq!(0,0,1)];
    assign_seq(&mut w_test_2, &seq_2_0, &index(&test_weights_cutouts,&seq_2_0));
    let seq_2_1 = [seq!(0,2,1),seq!(0,3,1),seq!(1,1,1)];
    assign_seq(&mut w_test_2, &seq_2_1, &index(&test_weights_cutouts,&seq_2_1));
    let seq_2_3 = [seq!(0,4,1),seq!(0,3,1),seq!(3,3,1)];
    assign_seq(&mut w_test_2, &seq_2_3, &index(&test_weights_cutouts,&seq_2_3));
    let seq_2_4 = [seq!(0,2,1),seq!(0,3,1),seq!(4,4,1)];
    assign_seq(&mut w_test_2, &seq_2_4, &index(&test_weights_cutouts,&seq_2_4));
    let expected = w_test_2.mul(network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_3 : Array<f64> = constant(0.0, dim4!(4,5,4,1));
    let seq_3_0 = [seq!(0,2,1),seq!(0,4,1),seq!(0,0,1)];
    assign_seq(&mut w_test_3, &seq_3_0, &index(&test_weights_cutouts,&seq_3_0));
    let seq_3_1 = [seq!(0,3,1),seq!(0,4,1),seq!(1,1,1)];
    assign_seq(&mut w_test_3, &seq_3_1, &index(&test_weights_cutouts,&seq_3_1));
    let seq_3_3 = [seq!(0,2,1),seq!(0,4,1),seq!(3,3,1)];
    assign_seq(&mut w_test_3, &seq_3_3, &index(&test_weights_cutouts,&seq_3_3));
    let expected = w_test_3.mul(network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_4 : Array<f64> = constant(0.0, dim4!(5,3,3,1));
    let seq_4_0 = [seq!(0,3,1),seq!(0,2,1),seq!(0,0,1)];
    assign_seq(&mut w_test_4, &seq_4_0, &index(&test_weights_cutouts,&seq_4_0));
    let seq_4_1 = [seq!(0,4,1),seq!(0,2,1),seq!(1,1,1)];
    assign_seq(&mut w_test_4, &seq_4_1, &index(&test_weights_cutouts,&seq_4_1));
    let expected = w_test_4.mul(network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());
}

#[test]
fn create_weights_3_3(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(3,3));

    let network = Network::new(Node::new(
        Uuid::new_v4(),
        NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 3),
                (attribute.clone(), 4),
                (attribute.clone(), 5),
                (attribute.clone(), 3)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(dim4!(1,2,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let w_dims = dim4!(5,5,7,5);
    let test_weights_cutouts : Array<f64> = constant(1.0, w_dims);

    let mut w_test_0 : Array<f64> = constant(0.0, dim4!(5,2,4,1));
    let seq_0_4 = [seq!(0,2,1),seq!(0,1,1),seq!(1,1,1)];
    assign_seq(&mut w_test_0, &seq_0_4, &index(&test_weights_cutouts, &seq_0_4));
    let seq_0_5 = [seq!(0,3,1),seq!(0,1,1),seq!(2,2,1)];
    assign_seq(&mut w_test_0, &seq_0_5, &index(&test_weights_cutouts, &seq_0_5));
    let seq_0_6 = [seq!(0,4,1),seq!(0,1,1),seq!(3,3,1)];
    assign_seq(&mut w_test_0, &seq_0_6, &index(&test_weights_cutouts, &seq_0_6));
    let expected = w_test_0.mul(network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_1 : Array<f64> = constant(0.0, dim4!(5,3,5,1));
    let seq_1_2 = [seq!(0,1,1),seq!(0,2,1),seq!(0,0,1)];
    assign_seq(&mut w_test_1, &seq_1_2, &index(&test_weights_cutouts,&seq_1_2));
    let seq_1_4 = [seq!(0,3,1),seq!(0,2,1),seq!(2,2,1)];
    assign_seq(&mut w_test_1, &seq_1_4, &index(&test_weights_cutouts,&seq_1_4));
    let seq_1_5 = [seq!(0,4,1),seq!(0,2,1),seq!(3,3,1)];
    assign_seq(&mut w_test_1, &seq_1_5, &index(&test_weights_cutouts,&seq_1_5));
    let seq_1_6 = [seq!(0,2,1),seq!(0,2,1),seq!(4,4,1)];
    assign_seq(&mut w_test_1, &seq_1_6, &index(&test_weights_cutouts,&seq_1_6));
    let expected = w_test_1.mul(network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_2 : Array<f64> = constant(0.0, dim4!(5,4,5,1));
    let seq_2_1 = [seq!(0,1,1),seq!(0,3,1),seq!(0,0,1)];
    assign_seq(&mut w_test_2, &seq_2_1, &index(&test_weights_cutouts,&seq_2_1));
    let seq_2_2 = [seq!(0,2,1),seq!(0,3,1),seq!(1,1,1)];
    assign_seq(&mut w_test_2, &seq_2_2, &index(&test_weights_cutouts,&seq_2_2));
    let seq_2_4 = [seq!(0,4,1),seq!(0,3,1),seq!(3,3,1)];
    assign_seq(&mut w_test_2, &seq_2_4, &index(&test_weights_cutouts,&seq_2_4));
    let seq_2_5 = [seq!(0,2,1),seq!(0,3,1),seq!(4,4,1)];
    assign_seq(&mut w_test_2, &seq_2_5, &index(&test_weights_cutouts,&seq_2_5));
    let expected = w_test_2.mul(network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_3 : Array<f64> = constant(0.0, dim4!(4,5,5,1));
    let seq_3_0 = [seq!(0,1,1),seq!(0,4,1),seq!(0,0,1)];
    assign_seq(&mut w_test_3, &seq_3_0, &index(&test_weights_cutouts,&seq_3_0));
    let seq_3_1 = [seq!(0,2,1),seq!(0,4,1),seq!(1,1,1)];
    assign_seq(&mut w_test_3, &seq_3_1, &index(&test_weights_cutouts,&seq_3_1));
    let seq_3_2 = [seq!(0,3,1),seq!(0,4,1),seq!(2,2,1)];
    assign_seq(&mut w_test_3, &seq_3_2, &index(&test_weights_cutouts,&seq_3_2));
    let seq_3_4 = [seq!(0,2,1),seq!(0,4,1),seq!(4,4,1)];
    assign_seq(&mut w_test_3, &seq_3_4, &index(&test_weights_cutouts,&seq_3_4));
    let expected = w_test_3.mul(network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());

    let mut w_test_4 : Array<f64> = constant(0.0, dim4!(5,3,4,1));
    let seq_4_0 = [seq!(0,2,1),seq!(0,2,1),seq!(0,0,1)];
    assign_seq(&mut w_test_4, &seq_4_0, &index(&test_weights_cutouts,&seq_4_0));
    let seq_4_1 = [seq!(0,3,1),seq!(0,2,1),seq!(1,1,1)];
    assign_seq(&mut w_test_4, &seq_4_1, &index(&test_weights_cutouts,&seq_4_1));
    let seq_4_2 = [seq!(0,4,1),seq!(0,2,1),seq!(2,2,1)];
    assign_seq(&mut w_test_4, &seq_4_2, &index(&test_weights_cutouts,&seq_4_2));
    let expected = w_test_4.mul(network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap());
    assert_eq!(&bincode::serialize(&expected).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_WEIGHTS").unwrap()).unwrap());
}