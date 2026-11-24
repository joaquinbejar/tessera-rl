// SPDX-License-Identifier: MIT OR Apache-2.0

//! `make_host` (ADR-0065 item 3), with the factory registered by the glue
//! rlib this binary references.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::{ConfigDecodeError, ConfigItem, ConfigSource, HostEnvironment};
use tessera_rl_engine::{
    ActionInput, DynHostPool, EngineError, FactoryError, HostPool, MakeOptions, Outputs,
    PoolOptions, Reseed, SyncOptions, make_host,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_test_host_env::{COUNTER_ID, Counter, CounterConfig};

fn pool_options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(6).unwrap(), Seed::new(21))
}

fn bytes(out: &Outputs) -> Vec<u8> {
    let mut v = out.observation(0).unwrap().to_vec();
    v.extend(out.rewards().iter().flat_map(|r| r.to_bits().to_le_bytes()));
    v.extend(out.episode_terminated().iter().map(|f| u8::from(f.get())));
    v.extend(out.pending_reset().iter().map(|f| u8::from(f.get())));
    v
}

fn trajectory(pool: &mut dyn DynHostPool) -> Vec<Vec<u8>> {
    let mut t = vec![bytes(pool.reset_all(Reseed::Keep).unwrap())];
    for k in 0..9 {
        let actions: Vec<i32> = (0..12).map(|i| (i + k) % 3).collect();
        t.push(bytes(
            pool.step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap(),
        ));
    }
    t
}

#[test]
fn test_factory_glue_rlib_is_linked() {
    assert_eq!(tessera_rl_test_host_glue::linked(), COUNTER_ID);
}

#[test]
fn test_factory_make_host_matches_host_pool_new_bit_for_bit() {
    let entries = [
        ("offset", ConfigItem::Int(-3)),
        ("reward", ConfigItem::Float(0.25)),
    ];
    let src = ConfigSource::new(&entries);
    for workers in [0, 2] {
        // The pool keeps its configuration (ADR-0073): one per pool.
        let cfg = CounterConfig {
            offset: -3,
            reward: 0.25,
            ..CounterConfig::default()
        };
        let mut options = MakeOptions::new(pool_options());
        let mut direct =
            HostPool::<Counter, { Counter::MAX_AGENTS }>::new(cfg, pool_options()).unwrap();
        if workers > 0 {
            let sync = SyncOptions::new(NonZeroUsize::new(workers).unwrap());
            options = options.with_sync(sync.clone());
            direct = direct.with_workers(sync).unwrap();
        }
        let mut made = make_host(COUNTER_ID, &src, &options).unwrap();
        assert_eq!(made.registration().id(), COUNTER_ID);
        assert_eq!(made.workers(), workers);
        assert_eq!(
            trajectory(&mut direct),
            trajectory(made.as_mut()),
            "workers={workers}"
        );
    }
}

#[test]
fn test_factory_errors() {
    let options = MakeOptions::new(pool_options());
    let empty = ConfigSource::empty();
    let err = |r: Result<Box<dyn DynHostPool>, FactoryError>| r.map(|_| ()).unwrap_err();
    assert_eq!(
        err(make_host("test/Nothing-v1", &empty, &options)),
        FactoryError::UnknownId {
            id: "test/Nothing-v1".to_owned()
        }
    );
    let no_factory = err(make_host("test/Unfactored-v1", &empty, &options));
    assert_eq!(
        no_factory,
        FactoryError::NoFactory {
            id: "test/Unfactored-v1"
        }
    );
    assert!(no_factory.to_string().contains("register_host_factory!"));
    let bad = [("colour", ConfigItem::Int(1))];
    assert_eq!(
        err(make_host(COUNTER_ID, &ConfigSource::new(&bad), &options)),
        FactoryError::Config(ConfigDecodeError::UnknownKey {
            key: "colour".to_owned()
        })
    );
    let invalid = [("episode", ConfigItem::Int(0))];
    assert!(matches!(
        err(make_host(
            COUNTER_ID,
            &ConfigSource::new(&invalid),
            &options
        )),
        FactoryError::Config(ConfigDecodeError::Invalid(_))
    ));
    // Bad worker options: an empty cpu set.
    let sync = SyncOptions {
        cpu_set: Some(Vec::new()),
        ..SyncOptions::new(NonZeroUsize::new(2).unwrap())
    };
    assert!(matches!(
        err(make_host(
            COUNTER_ID,
            &empty,
            &options.clone().with_sync(sync)
        )),
        FactoryError::Engine(EngineError::Options { .. })
    ));
}

#[test]
fn test_factory_env_crate_depends_on_core_only() {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let tree = std::process::Command::new(cargo)
        .args([
            "tree",
            "-q",
            "--offline",
            "-p",
            "tessera-rl-test-host-env",
            "-e",
            "normal",
            "--prefix",
            "none",
        ])
        .output()
        .unwrap();
    assert!(
        tree.status.success(),
        "{}",
        String::from_utf8_lossy(&tree.stderr)
    );
    let tree = String::from_utf8(tree.stdout).unwrap();
    assert!(tree.contains("tessera-rl-core "), "{tree}");
    assert!(!tree.contains("tessera-rl-engine"), "{tree}");
}

/// `register_host_factory!(E, snapshots)` (ADR-0073): a pool made by id
/// has snapshots through `DynHostPool`.
#[test]
fn test_factory_make_host_with_snapshots_round_trips() {
    let entries: [(&str, ConfigItem<'_>); 0] = [];
    let src = ConfigSource::new(&entries);
    let mut made = make_host(COUNTER_ID, &src, &MakeOptions::new(pool_options())).unwrap();
    assert!(made.snapshots() && made.layout().snapshots);
    made.reset_all(Reseed::Keep).unwrap();
    let records = made.snapshot_records(&[0]).unwrap();
    made.restore_records(&[1], &[&records[0]]).unwrap();
    assert_eq!(made.state_hash(1).unwrap(), made.state_hash(0).unwrap());
}
