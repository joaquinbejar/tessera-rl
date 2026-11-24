// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actor and critic: two MLPs, 4 -> 64 -> 64 -> {2, 1}, tanh.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// The policy and value networks.
#[derive(Module, Debug)]
pub struct ActorCritic<B: Backend> {
    a1: Linear<B>,
    a2: Linear<B>,
    a3: Linear<B>,
    c1: Linear<B>,
    c2: Linear<B>,
    c3: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    /// A new network on `device`.
    pub fn new(obs: usize, actions: usize, device: &B::Device) -> Self {
        let h = 64;
        Self {
            a1: LinearConfig::new(obs, h).init(device),
            a2: LinearConfig::new(h, h).init(device),
            a3: LinearConfig::new(h, actions).init(device),
            c1: LinearConfig::new(obs, h).init(device),
            c2: LinearConfig::new(h, h).init(device),
            c3: LinearConfig::new(h, 1).init(device),
        }
    }

    /// Action logits, `[batch, actions]`.
    pub fn logits(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.a1.forward(x).tanh();
        let h = self.a2.forward(h).tanh();
        self.a3.forward(h)
    }

    /// State values, `[batch]`.
    pub fn value(&self, x: Tensor<B, 2>) -> Tensor<B, 1> {
        let h = self.c1.forward(x).tanh();
        let h = self.c2.forward(h).tanh();
        self.c3.forward(h).squeeze(1)
    }
}
