/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use arrayfire::{Array, dim4};
use uuid::Uuid;
use crate::node::attribute::{Activation, Attribute, CellType, TensorDescriptor};
use crate::node::edges::{LinkType, NodeRange};
use crate::node::edges::LinkType::{InputsFromParent, OutputsToParent};
use crate::node::layer::Layer;
use crate::node::processor::processor::Processor;
use crate::node::network::Network;
use crate::node::node::{Node, NodeType};
use crate::node::scope::Scope;
use crate::node::tensor::Tensor;


fn tensor_sum(list: &Layer) -> usize {
    list.tensor.hash_map.iter().filter(|(k,_)| !k.contains("_SYSTEM")).map(|(_,v)| v.dims()[1] as usize).sum()
}

#[test]
fn create_links_r_w2m_all(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    //Create parent with one child
    let parent = Network::new(
        Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
        attribute.clone(),
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
            attribute.clone(),
            vec![
                (attribute.clone(), 3),
                (attribute.clone(), 2)
            ]
        )),
        attribute.clone()
    )));

    //Add new child to new row
    parent.node().write().unwrap().add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 3)
            ]
        )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));
    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));
    assert_eq!(parent.node().read().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));


    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));

    let inputs = Tensor::new(&[("input", Array::new_empty(dim4!(1,3,1,1)))]);
    parent.node().read().unwrap().traverse(&inputs, true);

    //Root
    assert_eq!(tensor_sum(&parent.node().read().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().mesh().layers()[1].lock().unwrap()), 5);

    //Child1
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 2);

    //Child2
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 3);
}

#[test]
fn create_links_r_w2m_selective(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    //Create parent with one child
    let parent = Network::new(
        Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                    attribute.clone(),
                    vec![
                        (attribute.clone(), 3),
                        (attribute.clone(), 2)
                    ]
                )),
            attribute.clone()
    )));

    //Add new child to new row
    parent.node().write().unwrap().add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 3)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(parent.node().read().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string()])), Ok(()));
    assert_eq!( parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input1".to_string()])), Ok(()));

    let input0 = ("input0",Array::new_empty(dim4!(1,2,1,1)));
    let input1 = ("input1",Array::new_empty(dim4!(1,1,1,1)));
    let inputs = Tensor::new(&[input0, input1]);
    parent.node().read().unwrap().traverse(&inputs, true);

    //Root
    assert_eq!(tensor_sum(&parent.node().read().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().mesh().layers()[1].lock().unwrap()), 5);

    //Child1
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 2);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 2);

    //Child2
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 1);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 3);
}

#[test]
fn create_links_r_w2m_wm1out0(){
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    //Create parent with one child
    let parent = Network::new(
        Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                    attribute.clone(),
                    vec![
                        (attribute.clone(), 3),
                        (attribute.clone(), 2)
                    ]
                )),
            attribute.clone()
    )));

    //Add new child to new row
    parent.node().write().unwrap().add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 3)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(parent.node().read().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(LinkType::InputsFromPeerOutputs((0,1)), NodeRange::All), Ok(()));

    let input0 = ("input0",Array::new_empty(dim4!(1,3,1,1)));
    let inputs = Tensor::new(&[input0]);
    parent.node().read().unwrap().traverse(&inputs, true);

    //Root
    assert_eq!(tensor_sum(&parent.node().read().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().mesh().layers()[1].lock().unwrap()), 5);

    //Child1
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 6);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 2);

    //Child2
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&parent.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 3);
}

#[test]
// A Parent network has two child networks, each child has a single layer of 2 meshes. no pass back.
fn create_links_r_w2n_w2m_all() {
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    // First Child
    let mut network_1_0 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                    attribute.clone(),
                    vec![
                        (attribute.clone(), 3),
                        (attribute.clone(), 2)
                    ]
                )),
            attribute.clone()
    ));

    network_1_0.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 4)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_0.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_0.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(network_1_0.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));

    // Second Child
    let mut network_1_1 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 2),
                    (attribute.clone(), 3)
                ]
            )),
            attribute.clone()
        ));

    network_1_1.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 4)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_1.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_1.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string()])), Ok(()));
    assert_eq!(network_1_1.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input1".to_string()])), Ok(()));

    // Parent Network
    let network_0_0 = Network::new(
        Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            network_1_0,
            attribute.clone()
        )));

    //Tell children to pass outputs to parent
    assert_eq!(network_0_0.node().read().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));


    network_0_0.node().write().unwrap().add_child_to_parent(network_1_1, Some(0)).unwrap();
    assert_eq!(network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string()])), Ok(()));
    assert_eq!(network_0_0.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));

    let input0 = ("input0",Array::new_empty(dim4!(1,2,1,1)));
    let input1 = ("input1",Array::new_empty(dim4!(1,3,1,1)));
    let inputs = Tensor::new(&[input0, input1]);
    network_0_0.node().read().unwrap().traverse(&inputs, true);

    // Root
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[0].lock().unwrap()), 5);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[1].lock().unwrap()), 13);
    //Root child 1 - Network 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 2);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 6);
    // Network 1 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 2);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 2);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 2);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 4);
    //Root child 2 - Network 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 5);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 7);
    // Network 2 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 2);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 3);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 4);
}

#[test]
// A Parent network has two child networks, each child has a single layer of 2 meshes. no pass back.
fn create_links_r_w2n_w2m_selective() {
    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    // First Child
    let mut network_1_0 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                    attribute.clone(),
                    vec![
                        (attribute.clone(), 3),
                        (attribute.clone(), 2)
                    ]
                )),
            attribute.clone()
    ));
    network_1_0.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 4)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_0.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_0.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(network_1_0.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));


    // Second Child
    let mut network_1_1 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                    attribute.clone(),
                    vec![
                        (attribute.clone(), 2),
                        (attribute.clone(), 3)
                    ]
                )),
        attribute.clone()
        ));

    network_1_1.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 4)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_1.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_1.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(network_1_1.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));

    // Parent Network
    let network_0_0 = Network::new(
        Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            network_1_0,
            attribute.clone()
    )));

    //Tell children to pass outputs to parent
    assert_eq!(network_0_0.node().read().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));
    network_0_0.node().write().unwrap().add_child_to_parent(network_1_1, Some(0)).unwrap();

    let input0 = ("input0",Array::new_empty(dim4!(1,3,1,1)));
    let inputs = Tensor::new(&[input0]);
    network_0_0.node().read().unwrap().traverse(&inputs, true);

    // Root
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[1].lock().unwrap()), 13);
    //Root child 1 - Network 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 6);
    // Network 1 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 2);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 4);
    //Root child 2 - Network 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 7);
    // Network 2 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 3);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,1),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 4);
}

#[test]
//A Parent network has two layers, each with a child network, each child has 2 meshes in on layer.
// Child network 1 receives outputs from child network 2's outputs.
fn create_links_r_w2n_w2m_wn1out0_all() {

    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    // First Child
    let mut network_1_0 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                    attribute.clone(),
                    vec![
                        (attribute.clone(), 3),
                        (attribute.clone(), 2)
                    ]
                )),
            attribute.clone()
    ));
    network_1_0.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 4)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_0.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_0.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(network_1_0.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    // The 1,0 is from the perspective of the parent.node().read().unwrap().
    // network_1_0 is 0,0 from the perspective of the parent.node().read().unwrap().
    // network_1_1 is 1,0 from the parent perspective.
    assert_eq!(network_1_0.mesh().edges().link(LinkType::InputsFromPeerOutputs((1,0)), NodeRange::All), Ok(()));

    // Second Child
    let mut network_1_1 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(
                Uuid::new_v4(),
                NodeType::Leaf(
                    attribute.clone(),
                    vec![
                        (attribute.clone(), 2),
                        (attribute.clone(), 3)
                    ]
                )),
            attribute.clone()
        ));

    network_1_1.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 4)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_1.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_1.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));
    assert_eq!(network_1_1.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::All), Ok(()));

    // Parent Network
    let network_0_0 = Network::new(
        Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            network_1_0,
            attribute.clone()
        )));
    assert_eq!(network_0_0.node().read().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));
    network_0_0.node().write().unwrap().add_child_to_parent(network_1_1, None).unwrap();

    let input0 = ("input0",Array::new_empty(dim4!(1,3,1,1)));
    let inputs = Tensor::new(&[input0]);
    network_0_0.node().read().unwrap().traverse(&inputs, true);

    // Root
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[1].lock().unwrap()), 13);
    //Root child 1 - Network 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 10);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 6);
    // Network 1 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 2);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 4);
    //Root child 2 - Network 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 7);
    // Network 2 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 3);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 4);
}


#[test]
//A Parent network has two layers, each with a child network, each child has 2 meshes in on layer.
// Child network 1 receives outputs from child network 2's outputs.
fn create_links_r_w2n_w2m_wn1out0_selective() {

    let attribute = Attribute::new(
        Activation::None,
        CellType::Mlp,
        vec![("_SYSTEM_WEIGHTS", TensorDescriptor::RandN), ("_SYSTEM_BIASES", TensorDescriptor::Const(1.0))],
        Scope::new(0,1));

    // First Child
    let mut network_1_0 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            Node::new(Uuid::new_v4(),
                      NodeType::Leaf(
                          attribute.clone(),
                          vec![
                              (attribute.clone(), 3),
                              (attribute.clone(), 1)
                          ]
                      )),
            attribute.clone()
        ));

    network_1_0.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 2)
                ]
        )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_0.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_0.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string()])), Ok(()));
    assert_eq!(network_1_0.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input1".to_string()])), Ok(()));

    // Second Child
    let mut network_1_1 = Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
        attribute.clone(),
        Node::new(Uuid::new_v4(),
                  NodeType::Leaf(
                      attribute.clone(),
                      vec![
                          (attribute.clone(), 2),
                          (attribute.clone(), 3)
                      ]
                  )),
        attribute.clone()
    ));
    network_1_1.add_child_to_parent(
        Node::new(
            Uuid::new_v4(),
            NodeType::Leaf(
                attribute.clone(),
                vec![
                    (attribute.clone(), 3),
                    (attribute.clone(), 4)
                ]
            )), Some(0)
    ).unwrap();

    //Tell children to pass outputs to parent
    assert_eq!(network_1_1.mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    assert_eq!(network_1_1.index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string()])), Ok(()));
    assert_eq!(network_1_1.index_into_node(&[(0,1)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input2".to_string()])), Ok(()));

    //Restrict the values passed to the parent
    let guid = network_1_1.index_into_node(&[(0,0)]).unwrap().lock().unwrap().uuid();
    assert_eq!(network_1_1.mesh().edges().link(OutputsToParent, NodeRange::Selective(vec![guid.to_string()])), Ok(()));

    // Parent Network
    let network_0_0 = Network::new(
        Node::new(
        Uuid::new_v4(),
        NodeType::Vertex(
            attribute.clone(),
            network_1_0,
            attribute.clone()
        )));

    //Tell children to pass outputs to parent
    assert_eq!(network_0_0.node().read().unwrap().mesh().edges().link(LinkType::OutputsToParent, NodeRange::All),Ok(()));

    network_0_0.node().write().unwrap().add_child_to_parent(network_1_1, None).unwrap();
    assert_eq!(network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string(),"input1".to_string()])), Ok(()));
    assert_eq!(network_0_0.node().read().unwrap().index_into_node(&[(1,0)]).unwrap().lock().unwrap().mesh().edges().link(InputsFromParent, NodeRange::Selective(vec!["input0".to_string(),"input2".to_string()])), Ok(()));

    //Assign sibling connection after both sibling are created, b/c UUID of output is needed.
    let guid = network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,1)]).unwrap().lock().unwrap().uuid();
    assert_eq!(network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().edges().link(LinkType::InputsFromPeerOutputs((1,0)), NodeRange::Selective(vec![guid.to_string()])), Ok(()));

    //Build
    let input0 = ("input0",Array::new_empty(dim4!(1,1,1,1)));
    let input1 = ("input1",Array::new_empty(dim4!(1,2,1,1)));
    let input2 = ("input2",Array::new_empty(dim4!(1,3,1,1)));
    let inputs = Tensor::new(&[input0, input1, input2]);
    network_0_0.node().read().unwrap().traverse(&inputs, true);

    // Root
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[0].lock().unwrap()), 6);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().mesh().layers()[1].lock().unwrap()), 6);
    //Root child 1 - Network 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 7);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 3);
    // Network 1 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 1);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 1);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 2);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(0,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 2);
    //Root child 2 - Network 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 4);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0)]).unwrap().lock().unwrap().mesh().layers()[1].lock().unwrap()), 7);
    // Network 2 - child 1
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 1);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,0)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 3);
    // Network 2 - child 2
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[0].lock().unwrap()), 3);
    assert_eq!(tensor_sum(&network_0_0.node().read().unwrap().index_into_node(&[(1,0),(0,1)]).unwrap().lock().unwrap().mesh().layers()[2].lock().unwrap()), 4);
}