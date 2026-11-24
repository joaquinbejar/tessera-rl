// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Renderer` runs `draw` and the rasterizer under the canonical
//! floating-point controls whatever the caller's (ADR-0083, REVIEW-4a1-r1):
//! a probe environment records `fp::is_canonical()` inside `draw`.

use core::sync::atomic::{AtomicBool, Ordering};

use tessera_rl_core::encoding::{Canonical, Decoder, Encoder, Sink};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, DecodeError, EnvConfig, EnvState, Environment, NoAdaptive, RecordHeader, SpaceError,
    StateInit, StepOut, Tier,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
struct Cell {
    value: u32,
}

impl Canonical for Cell {
    const ENCODED_LEN: usize = 4;

    fn encode<S: Sink>(&self, out: &mut Encoder<S>) {
        self.value.encode(out);
    }

    fn decode(input: &mut Decoder<'_>) -> Result<Self, DecodeError> {
        Ok(Self {
            value: Canonical::decode(input)?,
        })
    }
}

impl StateInit for Cell {
    fn state_default() -> Self {
        Self::default()
    }
}

impl EnvState for Cell {}

struct FixtureConfig;

impl EnvConfig for FixtureConfig {}

use tessera_rl_core::Render;
use tessera_rl_core::render::{Px16, Rgba, SceneWriter};
use tessera_rl_num::fp::{self, hostile};
use tessera_rl_render::Renderer;

/// Whether the last `draw` ran under the canonical controls.
static DRAWN_CANONICAL: AtomicBool = AtomicBool::new(false);

struct Probe;

impl Environment for Probe {
    type State = Cell;
    type Config = FixtureConfig;
    type Obs = ();
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(_: &FixtureConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(_: &FixtureConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &FixtureConfig, _: &mut StepRng, _: &mut Cell) {}

    fn step(
        _: &FixtureConfig,
        _: &mut Cell,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
    }

    fn observe(
        _: &FixtureConfig,
        _: &Cell,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, ()>,
    ) {
    }
}

impl Render for Probe {
    fn canvas(_: &FixtureConfig) -> (u16, u16) {
        (2, 2)
    }

    fn draw(_: &FixtureConfig, _: &Cell, scene: &mut SceneWriter<'_>) {
        DRAWN_CANONICAL.store(fp::is_canonical(), Ordering::SeqCst);
        scene.rect(
            Px16(0),
            Px16(0),
            Px16::px(2),
            Px16::px(2),
            Rgba::rgb(1, 2, 3),
        );
    }
}

#[test]
fn test_renderer_hostile_caller_controls_draw_canonical() {
    let mut renderer = Renderer::<Probe>::new(&FixtureConfig).unwrap();
    let mut frame = vec![0; renderer.frame_len()];
    let ((), _) = hostile::run(hostile::flush_to_zero, || {
        assert!(!fp::is_canonical(), "the hostile controls are set");
        renderer
            .frame_into(&FixtureConfig, &Cell::default(), &mut frame)
            .unwrap();
    });
    assert!(DRAWN_CANONICAL.load(Ordering::SeqCst));
    assert_eq!(frame, [1, 2, 3].repeat(4));
}
