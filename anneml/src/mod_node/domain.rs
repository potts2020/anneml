use crate::mod_node::scope::Scope;

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub (crate) struct Domain {
    underflow: usize,
    start: usize,
    index: usize,
    end: usize,
    overflow: usize
}

impl Domain {
    pub(crate) fn new(scope: &Scope, index: usize, layer_count: isize) -> Self {
        let underflow_start_diff = index as isize - scope.lower_bound() as isize;
        let overflow_end_diff = (index + scope.upper_bound()) as isize - (layer_count - 1);
        Domain {
            underflow: if underflow_start_diff < 0 { -underflow_start_diff as usize  } else { 0 },
            start: if underflow_start_diff > 0 { underflow_start_diff as usize} else { 0 },
            index,
            end: if overflow_end_diff > 0 { (layer_count - 1) as usize } else { index + scope.upper_bound() },
            overflow: if overflow_end_diff < 0 { 0 } else { overflow_end_diff as usize }
        }
    }

    pub(crate) fn underflow(&self) -> usize {
        self.underflow
    }
    pub(crate) fn start(&self) -> usize {
        self.start
    }
    pub(crate) fn index(&self) -> usize {
        self.index
    }
    pub(crate) fn end(&self) -> usize {
        self.end
    }
    pub(crate) fn overflow(&self) -> usize {
        self.overflow
    }
    pub(crate) fn domain_data(&self) -> (usize, usize, usize, usize, usize) {
        (self.underflow, self.start, self.index, self.end, self.overflow)
    }
}