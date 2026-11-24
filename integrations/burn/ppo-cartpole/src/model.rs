// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actor and critic: two MLPs, 4 -> 64 -> 64 -> {2, 1}, tanh.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::{Device, Module, Tensor};

/// The policy and value networks.
#[derive(Module, Debug)]
pub struct ActorCritic {
    a1: Linear,
    a2: Linear,
    a3: Linear,
    c1: Linear,
    c2: Linear,
    c3: Linear,
}

impl ActorCritic {
    /// A new network on `device`.
    pub fn new(obs: usize, actions: usize, device: &Device) -> Self {
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
    pub fn logits(&self, x: Tensor<2>) -> Tensor<2> {
        let h = self.a1.forward(x).tanh();
        let h = self.a2.forward(h).tanh();
        self.a3.forward(h)
    }

    /// State values, `[batch]`.
    pub fn value(&self, x: Tensor<2>) -> Tensor<1> {
        let h = self.c1.forward(x).tanh();
        let h = self.c2.forward(h).tanh();
        self.c3.forward(h).squeeze_dim::<1>(1)
    }
}
