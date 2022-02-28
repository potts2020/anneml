# Anneml

Artificial Neural Network and Extensible Machine Learning Library

## Another ML Library?

Anneml is a composite neural network builder designed with simplicity in mind. Composite neural networks are *networks mode of networks*. Anneml is designed such that these single networks can be trained independently and inserted into the larger system for efficient training and troubleshooting. Additionally, Anneml is designed with a **write once run anywhere** architecture.

## Fine-grain control

Anneml neural networks have layer level control for each individual network. The system is flexibly designed to allow for many node types in a single network at a layer level, minimizing surface complexity and extending the capabilities of each network.

## My First Network

Networks can be created using the following: 

```rust
//Instantiate attributes for a networks layer
let attribute = Attribute::new(
    Activation::None,// Activation Method
    CellType::Mlp,// Cell Type
    vec![
    	("_SYSTEM_WEIGHTS", TensorDescriptor::Const(0.3)), // Instantiate Weight Values
    	("_SYSTEM_BIASES", TensorDescriptor::Const(0.5))], // Instantiate Bias Values
    Scope::new(0,1)); // Determine the layer weight scope

//Assign attributes to a new neural network
    let mut network = Network::new(
        Node::new(
         	Uuid::from_u128(0), // Assign a Seed
            NodeType::Leaf(
                attribute.clone(), // Assign an input layer
                vec![
                    (attribute.clone(), 2), // Hidden layer
                    (attribute.clone(), 2), // Output layer
                ]
            )));
```



## Road Map

- Built In Back Propagation Evaluation
- Long Short Term Memory Cell Type
- Kernel Convolution Cell Type
- Synaptic Pruning for greater network efficiency
- Connecting to network clusters via local network

