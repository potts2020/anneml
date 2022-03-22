/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use arrayfire::{constant, dim4};
use uuid::Uuid;
use crate::node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
use crate::node::processor::processor::Processor;
use crate::node::network::Network;
use crate::node::node::{Node, NodeType};
use crate::node::scope::Scope;
use crate::node::tensor::Tensor;
use serial_test::serial;

#[test]
#[serial]
fn save_default() {

    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::Const(0.3)), ("_SYSTEM_BIASES", TensorDescriptor::Const(0.5))],
        Scope::new(0,1));


    let mut network = Network::new(
        Node::new(
            Uuid::from_u128(0),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 2),
                    (attribute.clone(), 2),
                ]
            )));

    let tensor = Tensor::new(&[("input", constant(1.0,dim4!(1,1,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    let expected : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                  0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 0, 154, 153, 153, 153, 153, 153,
                                  233, 63, 154, 153, 153, 153, 153, 153, 233, 63, 92, 143, 194, 245, 40, 92, 239, 63, 92, 143, 194, 245, 40, 92, 239, 63];
    assert_eq!(&expected, &bincode::serialize(&network.node().read().unwrap().mesh().tensor().hash_map.get("_SYSTEM_VALUES").unwrap()).unwrap());

    //Save the network.
    network.save();
}

#[test]
#[serial]
fn load_default() {
    let network = Network::load("00000000-0000-0000-0000-000000000000.annml").unwrap();
    let tensor = Tensor::new(&[("input", constant(1.0,dim4!(1,1,1,1)))]);
    network.node().read().unwrap().traverse(&tensor, true);

    let expected : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                  0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 0, 154, 153, 153, 153, 153, 153,
                                  233, 63, 154, 153, 153, 153, 153, 153, 233, 63, 92, 143, 194, 245, 40, 92, 239, 63, 92, 143, 194, 245, 40, 92, 239, 63];
    assert_eq!(&expected, &bincode::serialize(&network.node().read().unwrap().mesh().tensor().hash_map.get("_SYSTEM_VALUES").unwrap()).unwrap());}