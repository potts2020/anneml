/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use crate::node::attribute::{Attribute, TensorDescriptor};
use crate::node::domain::Domain;
use crate::node::tensor::Tensor;
use crate::node::utils::new_array;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Layer {
    pub(crate) tensor: Tensor,
    pub(crate) attribute: Attribute,
    pub(crate) domain: Domain
}

impl Layer {
    /// Layers contain Values and Attributes that modify how the layer transforms input values.
    /// # Return Values
    /// Layer
    pub(crate) fn new(attribute: Attribute, tensor: Tensor) -> Self {
        Layer { tensor, attribute, domain: Domain::default() }
    }

    /// Builds a layer that represents the connections to other layers in the slice.
    ///
    /// Indices represent the Neighboring Layers and their column size. contained inside.
    pub(crate) fn build(&mut self, index: usize, slice: &[usize]) {
        self.domain = Domain::new(self.attribute.scope(), index, slice.len() as isize);
        let (start, end) = (self.domain.start(), self.domain.end());
        let mut slice = slice[start..=end].to_vec();
        if slice.len() > 1 {
            self.tensor.insert("_SYSTEM_WEIGHTS", new_array(TypeTensor::Weight, &mut slice, &TensorDescriptor::Const(0.0), self.attribute.description("_SYSTEM_WEIGHTS"), start, index));
            self.tensor.insert("_SYSTEM_BIASES", new_array(TypeTensor::Bias, &mut slice, &TensorDescriptor::Const(0.0), self.attribute.description("_SYSTEM_BIASES"), start, index));
        }
    }


}

/// Flag that builds Arrayfire Arrays in a certain way.
pub(crate) enum TypeTensor {
    Weight,
    Bias,
}