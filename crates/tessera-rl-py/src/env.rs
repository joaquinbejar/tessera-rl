// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safe side of an environment object: one pool behind [`PoolOps`],
//! constructed from Python arguments. No
//! `unsafe` here; the C entry points live in `ffi`.

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_core::encoding::decode_from_slice;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{DecodeError, Environment, Validate};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, MAX_LEAVES, MaskProvider, Masked,
    NoMasks, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

/// Errors of the bindings: the engine's, plus record decoding and argument
/// errors detected here.
#[derive(Debug)]
pub(crate) enum BindError {
    /// An engine error.
    Engine(EngineError),
    /// A record that does not decode.
    Decode(DecodeError),
    /// An argument the bindings reject before reaching the engine.
    Argument(String),
}

impl From<EngineError> for BindError {
    fn from(e: EngineError) -> Self {
        Self::Engine(e)
    }
}

/// One observation leaf, as Python sees it.
#[derive(Debug, Clone)]
pub(crate) struct Leaf {
    /// Leaf name (empty for a single unnamed box).
    pub(crate) name: String,
    /// NumPy dtype string (`"float32"`, `"uint8"`, ...).
    pub(crate) dtype: &'static str,
    /// Shape of one agent's observation.
    pub(crate) shape: Vec<u32>,
    /// Inclusive bounds per element, as floats (integer bounds are exact
    /// up to 2^53; infinities for unbounded float elements).
    pub(crate) low: Vec<f64>,
    pub(crate) high: Vec<f64>,
}

/// How a reset chooses seeds.
#[derive(Debug, Clone)]
pub(crate) enum ResetSeed {
    /// Keep the seeds; the episode index advances.
    Keep,
    /// The k-th reset instance gets `base + k` (Gymnasium vector rule).
    Base(u64),
    /// The k-th reset instance gets `seeds[k]`.
    List(Vec<u64>),
}

/// The fixed layout of a pool's outputs.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub(crate) num_envs: usize,
    pub(crate) agents: usize,
    pub(crate) leaves: Vec<Leaf>,
    pub(crate) action_choices: u32,
    pub(crate) mask_len: usize,
    /// Terminal observations are written (`same_step` auto-reset).
    pub(crate) terminal: bool,
}

/// Copies of one call's outputs (`owned` mode).
#[derive(Debug, Clone)]
pub(crate) struct Owned {
    /// One buffer per leaf, `[N, A, leaf]`.
    pub(crate) observations: Vec<Vec<u8>>,
    /// `[N, A]`, little-endian f32.
    pub(crate) rewards: Vec<u8>,
    /// `[N, A]`, one byte per flag.
    pub(crate) agent_terminated: Vec<u8>,
    pub(crate) agent_truncated: Vec<u8>,
    pub(crate) alive: Vec<u8>,
    /// `[N]`.
    pub(crate) episode_terminated: Vec<u8>,
    pub(crate) episode_truncated: Vec<u8>,
    /// `[N, A, mask length]`; empty without masks.
    pub(crate) masks: Vec<u8>,
    /// One buffer per leaf in `same_step` mode, `[N, A, leaf]`: final
    /// observations of episodes that ended in the call (ADR-0039); rows of
    /// other instances are stale. Empty in other modes.
    pub(crate) terminal_observations: Vec<Vec<u8>>,
}

/// What an environment object can do, whatever the environment.
pub(crate) trait PoolOps: Send {
    fn layout(&self) -> &Layout;
    /// Resets every instance, or only `ids`.
    fn reset(&mut self, ids: Option<&[u32]>, seed: &ResetSeed) -> Result<(), BindError>;
    fn step(&mut self, actions: &[i32]) -> Result<(), BindError>;
    fn outputs(&self) -> Owned;
    /// The shared output storage, for borrowed views.
    fn storage(&self) -> std::sync::Arc<tessera_rl_engine::OutputStorage>;
    fn snapshot(&self, id: u32) -> Result<Vec<u8>, BindError>;
    fn import(&mut self, id: u32, bytes: &[u8]) -> Result<(), BindError>;
    fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), BindError>;
    fn is_poisoned(&self) -> bool;
}

struct Pooled<E: Environment, M: MaskProvider<E>> {
    pool: EnvPool<E, M>,
    layout: Layout,
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::F32 => "float32",
        DType::F64 => "float64",
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "bounds are exposed to Python as floats; documented on Leaf"
)]
fn float_bounds(bounds: &BoxBounds) -> (Vec<f64>, Vec<f64>) {
    match bounds {
        BoxBounds::Int { low, high } => (
            low.iter().map(|&v| v as f64).collect(),
            high.iter().map(|&v| v as f64).collect(),
        ),
        BoxBounds::UInt { low, high } => (
            low.iter().map(|&v| v as f64).collect(),
            high.iter().map(|&v| v as f64).collect(),
        ),
        BoxBounds::Float { low, high } => (low.clone(), high.clone()),
    }
}

fn leaves(space: &Space, name: &str, out: &mut Vec<Leaf>) -> Result<(), BindError> {
    match space {
        Space::Box {
            shape,
            dtype,
            bounds,
        } => {
            let (low, high) = float_bounds(bounds);
            out.push(Leaf {
                name: name.to_owned(),
                dtype: dtype_name(*dtype),
                shape: shape.clone(),
                low,
                high,
            });
        }
        Space::Dict(entries) => {
            for (key, inner) in entries {
                leaves(inner, key, out)?;
            }
        }
        _ => {
            return Err(BindError::Argument(
                "observation spaces other than boxes are not exposed yet".to_owned(),
            ));
        }
    }
    Ok(())
}

impl<E, M> Pooled<E, M>
where
    E: Environment + Validate,
    M: MaskProvider<E>,
{
    fn new(
        cfg: &E::Config,
        pool: EnvPool<E, M>,
        num_envs: usize,
        terminal: bool,
    ) -> Result<Self, BindError> {
        let mut obs = Vec::new();
        leaves(
            &E::observation_space(cfg).map_err(EngineError::from)?,
            "",
            &mut obs,
        )?;
        let Space::Discrete { n: action_choices } =
            E::action_space(cfg).map_err(EngineError::from)?
        else {
            return Err(BindError::Argument(
                "action spaces other than Discrete are not exposed yet".to_owned(),
            ));
        };
        let mask_len = if M::ENABLED {
            usize::try_from(action_choices).unwrap_or(0)
        } else {
            0
        };
        Ok(Self {
            pool,
            layout: Layout {
                num_envs,
                agents: E::MAX_AGENTS,
                leaves: obs,
                action_choices,
                mask_len,
                terminal,
            },
        })
    }
}

fn flags(v: &[tessera_rl_core::Bool8]) -> Vec<u8> {
    v.iter().map(|f| f.to_byte()).collect()
}

impl<E, M> PoolOps for Pooled<E, M>
where
    E: Environment + Validate,
    E::Config: Send + Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E> + Send,
{
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn reset(&mut self, ids: Option<&[u32]>, seed: &ResetSeed) -> Result<(), BindError> {
        let list: Vec<Seed>;
        let reseed = match seed {
            ResetSeed::Keep => Reseed::Keep,
            ResetSeed::Base(s) => Reseed::Base(Seed::new(*s)),
            ResetSeed::List(seeds) => {
                list = seeds.iter().map(|&s| Seed::new(s)).collect();
                Reseed::List(&list)
            }
        };
        match ids {
            None => self.pool.reset_all(reseed)?,
            Some(ids) => self.pool.reset(ids, reseed)?,
        };
        Ok(())
    }

    fn step(&mut self, actions: &[i32]) -> Result<(), BindError> {
        self.pool.step(ActionInput {
            discrete: actions,
            continuous: &[],
        })?;
        Ok(())
    }

    fn outputs(&self) -> Owned {
        let out = self.pool.outputs();
        let observations = (0..MAX_LEAVES)
            .map_while(|leaf| out.observation(leaf).map(<[u8]>::to_vec))
            .collect();
        let terminal_observations = if self.layout.terminal {
            (0..MAX_LEAVES)
                .map_while(|leaf| out.terminal_observation(leaf).map(<[u8]>::to_vec))
                .collect()
        } else {
            Vec::new()
        };
        Owned {
            terminal_observations,
            observations,
            rewards: out.rewards().iter().flat_map(|r| r.to_le_bytes()).collect(),
            agent_terminated: flags(out.agent_terminated()),
            agent_truncated: flags(out.agent_truncated()),
            alive: flags(out.alive()),
            episode_terminated: flags(out.episode_terminated()),
            episode_truncated: flags(out.episode_truncated()),
            masks: flags(out.masks()),
        }
    }

    fn storage(&self) -> std::sync::Arc<tessera_rl_engine::OutputStorage> {
        self.pool.outputs().storage()
    }

    fn snapshot(&self, id: u32) -> Result<Vec<u8>, BindError> {
        Ok(self.pool.snapshot(id)?.to_bytes())
    }

    fn import(&mut self, id: u32, bytes: &[u8]) -> Result<(), BindError> {
        let record = decode_from_slice::<DecodedRecord<E>>(bytes).map_err(BindError::Decode)?;
        self.pool.import(&[id], &[record])?;
        Ok(())
    }

    fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), BindError> {
        self.pool.fork(src, dst)?;
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        self.pool.is_poisoned()
    }
}

/// Pool arguments shared by every environment.
#[derive(Debug, Clone)]
pub(crate) struct PoolArgs {
    pub(crate) num_envs: u32,
    pub(crate) seed: u64,
    pub(crate) max_episode_steps: Option<u32>,
    pub(crate) auto_reset: String,
    pub(crate) chunk_size: Option<u32>,
    pub(crate) workers: usize,
}

fn options(args: &PoolArgs) -> Result<(PoolOptions, usize), BindError> {
    let num_envs = NonZeroU32::new(args.num_envs)
        .ok_or_else(|| BindError::Argument("num_envs must be at least 1".to_owned()))?;
    let auto_reset = match args.auto_reset.as_str() {
        "next_step" => AutoReset::NextStep,
        "same_step" => AutoReset::SameStep,
        "disabled" => AutoReset::Disabled,
        other => {
            return Err(BindError::Argument(format!(
                "auto_reset must be next_step, same_step or disabled, not {other:?}"
            )));
        }
    };
    let mut options = PoolOptions::new(num_envs, Seed::new(args.seed));
    options.auto_reset = auto_reset;
    options.max_episode_steps = args.max_episode_steps.and_then(NonZeroU32::new);
    if let Some(chunk) = args.chunk_size {
        options.chunk_size = NonZeroU32::new(chunk)
            .ok_or_else(|| BindError::Argument("chunk_size must be at least 1".to_owned()))?;
    }
    let n = usize::try_from(args.num_envs)
        .map_err(|_| BindError::Argument("num_envs too large".to_owned()))?;
    Ok((options, n))
}

fn with_workers<E, M>(pool: EnvPool<E, M>, workers: usize) -> Result<EnvPool<E, M>, BindError>
where
    E: Environment,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: MaskProvider<E>,
{
    match NonZeroUsize::new(workers) {
        Some(w) => Ok(pool.with_workers(SyncOptions::new(w))?),
        None => Ok(pool),
    }
}

/// A CartPole pool (`tessera/CartPole-v1`).
pub(crate) fn cartpole(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = CartPoleConfig::default();
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = with_workers(
        EnvPool::<CartPole, NoMasks>::new(cfg.clone(), options)?,
        args.workers,
    )?;
    Ok(Box::new(Pooled::new(&cfg, pool, n, same_step)?))
}

/// The Tag config fields, by name, as Python passes them.
#[derive(Debug, Clone, Default)]
pub(crate) struct TagFields {
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
    pub(crate) pursuers: Option<u32>,
    pub(crate) evaders: Option<u32>,
    pub(crate) slip: Option<u32>,
    pub(crate) radius: Option<u32>,
}

/// A Tag pool (`tessera/Tag-v1`), with masks.
pub(crate) fn tag(args: &PoolArgs, fields: &TagFields) -> Result<Box<dyn PoolOps>, BindError> {
    let d = TagConfig::default();
    let cfg = TagConfig {
        width: fields.width.unwrap_or(d.width),
        height: fields.height.unwrap_or(d.height),
        pursuers: fields.pursuers.unwrap_or(d.pursuers),
        evaders: fields.evaders.unwrap_or(d.evaders),
        slip: fields.slip.unwrap_or(d.slip),
        radius: fields.radius.unwrap_or(d.radius),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = with_workers(
        EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options)?,
        args.workers,
    )?;
    Ok(Box::new(Pooled::new(&cfg, pool, n, same_step)?))
}
