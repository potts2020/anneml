#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub(crate) struct Scope{
    lower_bound: usize,
    upper_bound: usize,
}

impl Scope {
    pub(crate) fn new(lower_bound: usize, upper_bound: usize) -> Self {
        Scope { lower_bound, upper_bound }
    }

    pub(crate) fn lower_bound(&self) -> usize {
        self.lower_bound
    }

    pub(crate) fn upper_bound(&self) -> usize {
        self.upper_bound
    }
}