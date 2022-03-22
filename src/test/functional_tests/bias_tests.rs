/*
 * Author:    Christian Potts
 * Created:   May 23rd, 2021
 *
 * (c) Copyright by Christian Potts
 */

/*
Test against the blueprint and ensures it works as expected
 */

use arrayfire::{Array, assign_seq, constant, Dim4, index, seq};
use uuid::Uuid;
use crate::node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
use crate::node::processor::processor::Processor;
use crate::node::network::Network;
use crate::node::node::{Node, NodeType};
use crate::node::scope::Scope;
use crate::node::tensor::Tensor;

#[test]
fn create_biases_0_1(){
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
                (attribute.clone(), 3),
                (attribute.clone(), 2),
                (attribute.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(Dim4::new(&[1,2,1,1])))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let test_biases_cutouts : Array<f64> = constant(1.0, Dim4::new(&[4,4,4,1]));

    let mut b_test_0 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_0_1 = [seq!(0,2,1),seq!(0,0,1),seq!(1,1,1)];
    assign_seq(&mut b_test_0, &seq_0_1, &index(&test_biases_cutouts, &seq_0_1));

    assert_eq!(&bincode::serialize(&b_test_0).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_1 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_1_1 = [seq!(0,3,1),seq!(0,0,1),seq!(1,1,1)];
    assign_seq(&mut b_test_1, &seq_1_1, &index(&test_biases_cutouts, &seq_1_1));
    assert_eq!(&bincode::serialize(&b_test_1).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_2 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_2_1 = [seq!(0,2,1),seq!(0,0,1),seq!(1,1,1)];
    assign_seq(&mut b_test_2, &seq_2_1, &index(&test_biases_cutouts, &seq_2_1));
    assert_eq!(&bincode::serialize(&b_test_2).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_3 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_3_1 = [seq!(0,1,1),seq!(0,0,1),seq!(1,1,1)];
    assign_seq(&mut b_test_3, &seq_3_1, &index(&test_biases_cutouts, &seq_3_1));
    assert_eq!(&bincode::serialize(&b_test_3).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_4 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_4_1 = [seq!(0,0,1),seq!(0,0,1),seq!(1,1,1)];
    assign_seq(&mut b_test_4, &seq_4_1, &index(&test_biases_cutouts, &seq_4_1));
    assert_eq!(&bincode::serialize(&b_test_4).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());
}

#[test]
fn create_biases_0_2(){
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
                (attribute.clone(), 3),
                (attribute.clone(), 2),
                (attribute.clone(), 1)
            ]
        )));


    let tensor = Tensor::new(&[("input", Array::new_empty(Dim4::new(&[1,2,1,1])))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let test_biases_cutouts : Array<f64> = constant(1.0, Dim4::new(&[4,4,4,1]));

    let z_seq_0 = seq!(1,1,1);
    let z_seq_1 = seq!(2,2,1);

    let mut b_test_0 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_0_1 = [seq!(0,2,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_0, &seq_0_1, &index(&test_biases_cutouts, &seq_0_1));
    let seq_0_2 = [seq!(0,3,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_0, &seq_0_2, &index(&test_biases_cutouts, &seq_0_2));
    assert_eq!(&bincode::serialize(&b_test_0).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_1 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_1_1 = [seq!(0,3,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_1, &seq_1_1, &index(&test_biases_cutouts, &seq_1_1));
    let seq_1_2 = [seq!(0,2,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_1, &seq_1_2, &index(&test_biases_cutouts, &seq_1_2));
    assert_eq!(&bincode::serialize(&b_test_1).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_2 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_2_1 = [seq!(0,2,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_2, &seq_2_1, &index(&test_biases_cutouts, &seq_2_1));
    let seq_2_2 = [seq!(0,1,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_2, &seq_2_2, &index(&test_biases_cutouts, &seq_2_2));
    assert_eq!(&bincode::serialize(&b_test_2).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_3 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_3_1 = [seq!(0,1,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_3, &seq_3_1, &index(&test_biases_cutouts, &seq_3_1));
    let seq_3_2 = [seq!(0,0,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_3, &seq_3_2, &index(&test_biases_cutouts, &seq_3_2));
    assert_eq!(&bincode::serialize(&b_test_3).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_4 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_4_1 = [seq!(0,0,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_4, &seq_4_1, &index(&test_biases_cutouts, &seq_4_1));
    assert_eq!(&bincode::serialize(&b_test_4).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());
}

#[test]
fn create_biases_1_1(){
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
                (attribute.clone(), 3),
                (attribute.clone(), 2),
                (attribute.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(Dim4::new(&[1,2,1,1])))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let test_biases_cutouts : Array<f64> = constant(1.0, Dim4::new(&[4,4,4,1]));

    let z_seq_0 = seq!(0,0,1);
    let z_seq_1 = seq!(2,2,1);

    let mut b_test_0 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_0_2 = [seq!(0,2,1),seq!(0,0,1),seq!(1,1,1)];
    assign_seq(&mut b_test_0, &seq_0_2, &index(&test_biases_cutouts, &seq_0_2));
    assert_eq!(&bincode::serialize(&b_test_0).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_1 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_1_0 = [seq!(0,1,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_1, &seq_1_0, &index(&test_biases_cutouts, &seq_1_0));
    let seq_1_2 = [seq!(0,3,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_1, &seq_1_2, &index(&test_biases_cutouts, &seq_1_2));
    assert_eq!(&bincode::serialize(&b_test_1).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_2 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_2_0 = [seq!(0,2,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_2, &seq_2_0, &index(&test_biases_cutouts, &seq_2_0));
    let seq_2_2 = [seq!(0,2,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_2, &seq_2_2, &index(&test_biases_cutouts, &seq_2_2));
    assert_eq!(&bincode::serialize(&b_test_2).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_3 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_3_0 = [seq!(0,3,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_3, &seq_3_0, &index(&test_biases_cutouts, &seq_3_0));
    let seq_3_2 = [seq!(0,1,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_3, &seq_3_2, &index(&test_biases_cutouts, &seq_3_2));
    assert_eq!(&bincode::serialize(&b_test_3).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_4 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_4_0 = [seq!(0,2,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_4, &seq_4_0, &index(&test_biases_cutouts, &seq_4_0));
    let seq_4_2 = [seq!(0,0,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_4, &seq_4_2, &index(&test_biases_cutouts, &seq_4_2));
    assert_eq!(&bincode::serialize(&b_test_4).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_5 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[5].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_5_0 = [seq!(0,1,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_5, &seq_5_0, &index(&test_biases_cutouts, &seq_5_0));
    assert_eq!(&bincode::serialize(&b_test_5).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[5].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());
}

#[test]
fn create_biases_2_2(){
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
                (attribute.clone(), 3),
                (attribute.clone(), 2),
                (attribute.clone(), 1)
            ]
        )));

    let tensor = Tensor::new(&[("input", Array::new_empty(Dim4::new(&[1,2,1,1])))]);
    network.node().read().unwrap().traverse(&tensor, true);
    let test_biases_cutouts : Array<f64> = constant(1.0, Dim4::new(&[5,5,5,1]));

    let z_seq_0 = seq!(0,0,1);
    let z_seq_1 = seq!(1,1,1);
    let z_seq_3 = seq!(3,3,1);
    let z_seq_4 = seq!(4,4,1);

    let mut b_test_0 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_0_3 = [seq!(0,2,1),seq!(0,0,1),seq!(1,1,1)];
    assign_seq(&mut b_test_0, &seq_0_3, &index(&test_biases_cutouts, &seq_0_3));
    let seq_0_4 = [seq!(0,3,1),seq!(0,0,1),seq!(2,2,1)];
    assign_seq(&mut b_test_0, &seq_0_4, &index(&test_biases_cutouts, &seq_0_4));
    assert_eq!(&bincode::serialize(&b_test_0).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[0].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_1 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_1_1 = [seq!(0,1,1),seq!(0,0,1),seq!(0,0,1)];
    assign_seq(&mut b_test_1, &seq_1_1, &index(&test_biases_cutouts, &seq_1_1));
    let seq_1_3 = [seq!(0,3,1),seq!(0,0,1),seq!(2,2,1)];
    assign_seq(&mut b_test_1, &seq_1_3, &index(&test_biases_cutouts, &seq_1_3));
    let seq_1_4 = [seq!(0,2,1),seq!(0,0,1),seq!(3,3,1)];
    assign_seq(&mut b_test_1, &seq_1_4, &index(&test_biases_cutouts, &seq_1_4));
    assert_eq!(&bincode::serialize(&b_test_1).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[1].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_2 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_2_0 = [seq!(0,1,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_2, &seq_2_0, &index(&test_biases_cutouts, &seq_2_0));
    let seq_2_1 = [seq!(0,2,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_2, &seq_2_1, &index(&test_biases_cutouts, &seq_2_1));
    let seq_2_3 = [seq!(0,2,1),seq!(0,0,1),z_seq_3];
    assign_seq(&mut b_test_2, &seq_2_3, &index(&test_biases_cutouts, &seq_2_3));
    let seq_2_4 = [seq!(0,1,1),seq!(0,0,1),seq!(4,4,1)];
    assign_seq(&mut b_test_2, &seq_2_4, &index(&test_biases_cutouts, &seq_2_4));
    assert_eq!(&bincode::serialize(&b_test_2).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[2].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_3 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_3_0 = [seq!(0,2,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_3, &seq_3_0, &index(&test_biases_cutouts, &seq_3_0));
    let seq_3_1 = [seq!(0,3,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_3, &seq_3_1, &index(&test_biases_cutouts, &seq_3_1));
    let seq_3_3 = [seq!(0,1,1),seq!(0,0,1),z_seq_3];
    assign_seq(&mut b_test_3, &seq_3_3, &index(&test_biases_cutouts, &seq_3_3));
    let seq_3_4 = [seq!(0,0,1),seq!(0,0,1),z_seq_4];
    assign_seq(&mut b_test_3, &seq_3_4, &index(&test_biases_cutouts, &seq_3_4));
    assert_eq!(&bincode::serialize(&b_test_3).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[3].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_4 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_4_0 = [seq!(0,3,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_4, &seq_4_0, &index(&test_biases_cutouts, &seq_4_0));
    let seq_4_1 = [seq!(0,2,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_4, &seq_4_1, &index(&test_biases_cutouts, &seq_4_1));
    let seq_4_3 = [seq!(0,0,1),seq!(0,0,1),z_seq_3];
    assign_seq(&mut b_test_4, &seq_4_3, &index(&test_biases_cutouts, &seq_4_3));
    assert_eq!(&bincode::serialize(&b_test_4).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[4].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());

    let mut b_test_5 : Array<f64> = constant(0.0, network.node().read().unwrap().mesh().layers()[5].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap().dims());
    let seq_5_0 = [seq!(0,2,1),seq!(0,0,1),z_seq_0];
    assign_seq(&mut b_test_5, &seq_5_0, &index(&test_biases_cutouts, &seq_5_0));
    let seq_5_1 = [seq!(0,1,1),seq!(0,0,1),z_seq_1];
    assign_seq(&mut b_test_5, &seq_5_1, &index(&test_biases_cutouts, &seq_5_1));
    assert_eq!(&bincode::serialize(&b_test_5).unwrap(),
               &bincode::serialize(&network.node().read().unwrap().mesh().layers()[5].lock().unwrap().tensor.hash_map.get("_SYSTEM_BIASES").unwrap()).unwrap());
}