#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Edges {
    pub(crate) input_from_parent: NodeRange,
    pub(crate) output_to_parent: NodeRange,
    pub(crate) input_from_peer_output: Vec<((usize, usize), NodeRange)>,
}

impl Edges {
    pub(crate) fn new(output_to_parent: NodeRange) -> Self {
        Edges { input_from_parent: NodeRange::All, output_to_parent, input_from_peer_output: vec![] }
    }

    pub(crate) fn link(&mut self, link: LinkType, node_range: NodeRange) -> Result<(), &'static str>{
        match link {
            LinkType::InputsFromPeerOutputs((x_from, y_from)) => { self.input_from_peer_output.push( ((x_from, y_from), node_range)); }
            LinkType::InputsFromParent => { self.input_from_parent = node_range; }
            LinkType::OutputsToParent => { self.output_to_parent = node_range; }
        }
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) enum LinkType {
    InputsFromPeerOutputs((usize, usize)),
    InputsFromParent,
    OutputsToParent,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(crate) enum NodeRange {
    Selective(Vec<String>),
    All,
}