/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Edges {
    pub(crate) input_from_parent: NodeRange,
    pub(crate) output_to_parent: NodeRange,
    pub(crate) input_from_peer_output: Vec<((usize, usize), NodeRange)>,
}

impl Edges {
    /// Logical Connections between Nodes.
    ///
    /// All Nodes pass their values to a parent, so outputs_to_parent are non-optional.
    ///
    /// # Return Values
    /// Edges
    pub(crate) fn new(output_to_parent: NodeRange) -> Self {
        Edges { input_from_parent: NodeRange::All, output_to_parent, input_from_peer_output: vec![] }
    }

    /// Assign a logical link to between Nodes.
    ///
    /// Link is of type LinkType and denotes if a link is to a Sibling (assign to peer input), to a parent (assign to output), or from a parent (assign to input).
    ///
    /// # Return Values
    /// Ok(())
    pub fn link(&mut self, link: LinkType, node_range: NodeRange) -> Result<(), &'static str>{
        match link {
            LinkType::InputsFromPeerOutputs((x_from, y_from)) => { self.input_from_peer_output.push( ((x_from, y_from), node_range)); }
            LinkType::InputsFromParent => { self.input_from_parent = node_range; }
            LinkType::OutputsToParent => { self.output_to_parent = node_range; }
        }
        Ok(())
    }
}

/// LinkType flag used in Edges link function.
#[derive(serde::Serialize, serde::Deserialize)]
pub enum LinkType {
    InputsFromPeerOutputs((usize, usize)),
    InputsFromParent,
    OutputsToParent,
}

/// Establishes if all values from a Node are to be linked if only certain values will be passed.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub enum NodeRange {
    Selective(Vec<String>),
    All,
}