/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
pub struct Scope{
    lower_bound: usize,
    upper_bound: usize,
}

impl Scope {
    pub fn new(lower_bound: usize, upper_bound: usize) -> Self {
        Scope { lower_bound, upper_bound }
    }

    pub(crate) fn lower_bound(&self) -> usize {
        self.lower_bound
    }

    pub(crate) fn upper_bound(&self) -> usize {
        self.upper_bound
    }
}