use crate::mod_node::attribute::{Attribute, TensorDescriptor};
use crate::mod_node::domain::Domain;
use crate::mod_node::tensor::Tensor;
use crate::mod_node::utils::new_array;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Layer {
    pub(crate) tensor: Tensor,
    pub(crate) attribute: Attribute,
    pub(crate) domain: Domain
}

impl Layer {
    pub(crate) fn new(attribute: Attribute, tensor: Tensor) -> Layer {
        Layer { tensor, attribute, domain: Domain::default() }
    }

    pub(crate) fn build(&mut self, index: usize, slice: &[usize]) {
        self.domain = Domain::new(self.attribute.scope(), index, slice.len() as isize);
        let (start, end) = (self.domain.start(), self.domain.end());
        let mut slice = slice[start..=end].to_vec();
        if slice.len() > 1 {
            self.tensor.insert("_SYSTEM_WEIGHTS", new_array(TypeTensor::Weight, &mut slice, &TensorDescriptor::Const(0.0), self.attribute.descriptions("_SYSTEM_WEIGHTS"), start, index));
            self.tensor.insert("_SYSTEM_BIASES", new_array(TypeTensor::Bias, &mut slice, &TensorDescriptor::Const(0.0), self.attribute.descriptions("_SYSTEM_BIASES"), start, index));
        }
    }


}

pub(crate) enum TypeTensor {
    Weight,
    Bias,
}