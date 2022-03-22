/*
 * Author:    Christian Potts
 * Created:   March 13th, 2022
 *
 * (c) Copyright by Christian Potts
 */

use std::mem;
use std::ops::Mul;
use arrayfire::{Array, assign_seq, constant, Dim4, dim4, randn, randu, seq};
use rustc_hash::FxHashMap;
use crate::node::attribute::TensorDescriptor;
use crate::node::layer::TypeTensor;

pub(crate) fn build_array(op: &TensorDescriptor, dims: arrayfire::Dim4) -> Array<f64> {
    match &op {
        TensorDescriptor::RandN =>  { randn(dims) }
        TensorDescriptor::RandU =>  { randu(dims) }
        TensorDescriptor::RangeN(rng)  => { (randn(dims) as Array<f64>).mul(*rng)}
        TensorDescriptor::RangeU(rng) => { (randu(dims) as Array<f64>).mul(*rng)}
        TensorDescriptor::Const(cnst) => { constant(*cnst, dims)}
        _ => { constant(0.0, dims) }
    }
}

pub(crate) fn new_array(array_type: TypeTensor, vec: &mut Vec<usize>, base: &TensorDescriptor,
                        op: &TensorDescriptor, start: usize, index: usize) -> Array<f64>{
    let mut xx = 0;
    match &array_type{
        TypeTensor::Weight => { mem::swap(&mut vec[(index - start) as usize], &mut xx); }
        TypeTensor::Bias => { xx = 1; vec[(index - start) as usize] = 0; }
    }
    let dims = Dim4::new(&[*vec.iter().max().unwrap() as u64,xx as u64,vec.len() as u64,1]);
    let mut array = build_array(base, dims);
    vec.iter().enumerate().filter(|e| *e.1 > 0).for_each(|e| {
        let seq = &[seq!(0,(*e.1 - 1) as i32,1), seq!(0,(xx - 1) as i32,1), seq!(e.0 as i32,e.0 as i32,1)];
        assign_seq(&mut array, seq , &build_array(op, dim4!(*e.1 as u64,xx as u64,1,1)));
    });
    array.eval();
    array
}


pub(crate) fn fx_hash_map_new<K: std::hash::Hash + std::cmp::Eq,V>(key_value_pair: Vec<(K, V)>) -> FxHashMap<K, V>{
    let mut hash_map = FxHashMap::default();
    for (k, v) in key_value_pair { hash_map.insert(k, v); }
    hash_map
}