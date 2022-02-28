use arrayfire::{constant, Dim4};
use uuid::Uuid;
use crate::mod_node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
use crate::mod_node::edges::LinkType::InputsFromParent;
use crate::mod_node::edges::NodeRange;
use crate::mod_node::mod_processor::processor::Processor;
use crate::mod_node::network::Network;
use crate::mod_node::node::{Node, NodeType};
use crate::mod_node::scope::Scope;
use crate::mod_node::tensor::Tensor;

#[test]
fn feed_forward_1x1_122_no_activation() {
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::Const(0.3)), ("_SYSTEM_BIASES", TensorDescriptor::Const(0.5))],
        Scope::new(0,1));

    let network = Network::new(
        Node::new(
            Uuid::new_v4(),
        NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 2),
                (attribute.clone(), 2),
            ]
        )));

    let tensor = Tensor::new(&[("input", constant(1.0,Dim4::new(&[1,1,1,1])))]);
    network.node().read().unwrap().traverse(&tensor, true);

    //not working because you are testing against the whole value array, not just the output.
    let expected : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 0, 154, 153, 153, 153, 153, 153,
        233, 63, 154, 153, 153, 153, 153, 153, 233, 63, 92, 143, 194, 245, 40, 92, 239, 63, 92, 143, 194, 245, 40, 92, 239, 63];
    assert_eq!(&expected, &bincode::serialize(&network.node().read().unwrap().mesh().tensor().hash_map.get("_SYSTEM_VALUES").unwrap()).unwrap());
}

#[test]
fn feed_forward_1x1_21_no_activation(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::Const(0.3)), ("_SYSTEM_BIASES", TensorDescriptor::Const(0.5))],
        Scope::new(0,1));

    let network = Network::new(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 1),
                ]
            )));

    let tensor = Tensor::new(&[("input", constant(1.0,Dim4::new(&[1,2,1,1])))]);
    network.node().read().unwrap().traverse(&tensor, true);

    let expected : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                  1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0,
                                  0, 0, 0, 240, 63, 154, 153, 153, 153, 153, 153, 241, 63, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(&expected, &bincode::serialize(&network.node().read().unwrap().mesh().tensor().hash_map.get("_SYSTEM_VALUES").unwrap()).unwrap());
}

#[test]
fn feed_forward_1x1_21_sigmoid(){
    let attribute = Attribute::new(
        Activation::Sigmoid,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::Const(0.3)), ("_SYSTEM_BIASES", TensorDescriptor::Const(0.5))],
        Scope::new(0,1));

    let network = Network::new(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 1),
                ]
            )));

    let tensor = Tensor::new(&[("input", constant(1.0,Dim4::new(&[1,2,1,1])))]);
    network.node().read().unwrap().traverse(&tensor, true);

    let expected : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                  0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 189, 162, 213, 245, 212, 100, 231, 63, 189, 162,
                                  213, 245, 212, 100, 231, 63, 159, 129, 154, 240, 154, 0, 231, 63, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(&expected, &bincode::serialize(&network.node().read().unwrap().mesh().tensor().hash_map.get("_SYSTEM_VALUES").unwrap()).unwrap());
}

#[test]
fn feed_forward_1x2_122_21_no_activation(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::Const(0.3)), ("_SYSTEM_BIASES", TensorDescriptor::Const(0.5))],
        Scope::new(0,1));

    //Create parent with one child
    let network = Network::new(
        Node::new(
            Uuid::new_v4(),
            NodeType::Vertex(
                attribute.clone(),
                Node::new(
                    Uuid::new_v4(),
                    NodeType::Leaf(
                        attribute.clone(),
                        vec![
                            (attribute.clone(), 2),
                            (attribute.clone(), 2),
                        ]
                    )),
                attribute.clone()
            )));

    //Add new child to new row
    network.node().write().unwrap().add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 1),
                ]
            )),
        Some(0)
    );

    //Only pass some inputs to children
    let input0 = ("input0",constant(1.0,Dim4::new(&[1,1,1,1])));
    let input1 = ("input1",constant(1.0,Dim4::new(&[1,2,1,1])));

    assert_eq!(network.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string()])), Ok(()));
    assert_eq!(network.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input1".to_string()])), Ok(()));

    let tensor = Tensor::new(&[input0, input1]);
    network.node().read().unwrap().traverse(&tensor, true);

    let expected0 : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 0, 154, 153, 153, 153, 153, 153, 233, 63,
                                   154, 153, 153, 153, 153, 153, 233, 63, 92, 143, 194, 245, 40, 92, 239, 63, 92, 143, 194, 245, 40, 92, 239, 63];
    assert_eq!(&expected0, &bincode::serialize(&network.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().tensor().hash_map.get("_SYSTEM_VALUES").unwrap()).unwrap());

    let expected1 : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                                   1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0,
                                   0, 0, 0, 240, 63, 154, 153, 153, 153, 153, 153, 241, 63, 0, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(&expected1, &bincode::serialize(&network.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().tensor().hash_map.get("_SYSTEM_VALUES").unwrap()).unwrap());

    // let parent_expected : Vec<u8> = vec![2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
    //                                      0, 0, 0, 92, 143, 194, 245, 40, 92, 239, 63, 92, 143, 194, 245, 40, 92, 239, 63, 154, 153, 153, 153, 153, 153, 241, 63];
    // assert_eq!(&expected, &bincode::serialize(
    //     &join(1, )
    // ).unwrap());
    //
    // assert_eq!(&parent_expected, &bincode::serialize(&join(1, &parent.index(&[(0,0)]).unwrap().read().unwrap().output(), &parent.index(&[(0,1)]).unwrap().read().unwrap().output())).unwrap());
}