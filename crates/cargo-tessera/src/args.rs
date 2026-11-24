// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-line arguments, parsed by hand (no dependency, #351).

use std::path::PathBuf;

/// The most agent slots `new` scaffolds (`MAX_HOST_AGENTS` of the engine).
pub const MAX_AGENTS: u16 = 256;

/// A parsed command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Print the help of a subcommand, or the general help.
    Help(Topic),
    /// Print the version.
    Version,
    /// `cargo tessera new`.
    New(NewArgs),
    /// `cargo tessera check`.
    Check(CheckArgs),
    /// `cargo tessera bench`.
    Bench(BenchArgs),
}

/// What `--help` describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Topic {
    /// The tool.
    Main,
    /// `new`.
    New,
    /// `check`.
    Check,
    /// `bench`.
    Bench,
}

/// The turn structure of a scaffolded environment (ADR-0013).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Turns {
    /// Every alive agent acts every step.
    Simultaneous,
    /// One agent acts per step, in slot order.
    Sequential,
    /// A subset of the agents acts per step.
    Mixed,
}

impl Turns {
    /// The name the Python side reports (`env.turns`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Simultaneous => "simultaneous",
            Self::Sequential => "sequential",
            Self::Mixed => "mixed",
        }
    }
}

/// Arguments of `new`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewArgs {
    /// The crate name.
    pub name: String,
    /// The directory to create; `./<name>` by default.
    pub path: Option<PathBuf>,
    /// The environment id; derived from the name by default.
    pub id: Option<String>,
    /// Agent slots.
    pub agents: u16,
    /// Turn structure.
    pub turns: Turns,
    /// Implement `HostSnapshot`.
    pub snapshots: bool,
    /// Depend on the Tessera crates by path, at this repository checkout,
    /// instead of on the exact released version.
    pub tessera_path: Option<PathBuf>,
}

/// Flags passed through to every cargo invocation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CargoFlags {
    /// `--locked`.
    pub locked: bool,
    /// `--offline`.
    pub offline: bool,
}

impl CargoFlags {
    /// The flags as cargo arguments.
    #[must_use]
    pub fn args(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.locked {
            out.push("--locked");
        }
        if self.offline {
            out.push("--offline");
        }
        out
    }
}

/// Arguments of `check`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckArgs {
    /// Also build the wheel and run the Python adapter tests.
    pub python: bool,
    /// Record the golden hashes instead of comparing them.
    pub bless: bool,
    /// Install the wheel with `uv pip install` instead of `maturin develop`.
    pub uv: bool,
    /// Cargo flags.
    pub cargo: CargoFlags,
}

/// Arguments of `bench`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BenchArgs {
    /// Criterion's `--quick`: a seconds budget.
    pub quick: bool,
    /// Cargo flags.
    pub cargo: CargoFlags,
}

/// A command line that cannot be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArgsError {
    /// No subcommand.
    #[error("no subcommand: expected `new`, `check` or `bench` (see `cargo tessera --help`)")]
    NoCommand,
    /// An unknown subcommand.
    #[error("unknown subcommand `{0}`: expected `new`, `check` or `bench`")]
    UnknownCommand(String),
    /// An option the subcommand does not take.
    #[error("unknown option `{option}` for `{command}` (see `cargo tessera {command} --help`)")]
    UnknownOption {
        /// The subcommand.
        command: &'static str,
        /// The option.
        option: String,
    },
    /// An option without its value.
    #[error("`{0}` needs a value")]
    MissingValue(&'static str),
    /// A flag that takes no value was given one.
    #[error("`{0}` takes no value")]
    UnexpectedValue(&'static str),
    /// An option given twice.
    #[error("`{0}` is given twice")]
    Repeated(&'static str),
    /// `new` without a name.
    #[error("`new` needs the crate name: cargo tessera new <name>")]
    MissingName,
    /// A positional argument the subcommand does not take.
    #[error("unexpected argument `{0}`")]
    Unexpected(String),
    /// An option value that is not valid.
    #[error("invalid value `{value}` for `{option}`: {reason}")]
    InvalidValue {
        /// The option.
        option: &'static str,
        /// The value.
        value: String,
        /// Why.
        reason: &'static str,
    },
}

/// One argument: a flag with an optional inline value (`--x=v`), or a
/// positional.
enum Arg {
    Flag(String, Option<String>),
    Positional(String),
}

fn split(arg: String) -> Arg {
    if arg.starts_with("--") {
        match arg.split_once('=') {
            Some((flag, value)) => Arg::Flag(flag.to_owned(), Some(value.to_owned())),
            None => Arg::Flag(arg, None),
        }
    } else if arg == "-h" {
        Arg::Flag("--help".to_owned(), None)
    } else {
        Arg::Positional(arg)
    }
}

/// Parses the process arguments, the program name first. Cargo runs
/// `cargo-tessera tessera <args>`; the `tessera` word is skipped.
///
/// # Errors
///
/// An [`ArgsError`] for anything the tool does not accept.
pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Command, ArgsError> {
    let mut args = args.into_iter().skip(1).peekable();
    if args.peek().map(String::as_str) == Some("tessera") {
        args.next();
    }
    let Some(command) = args.next() else {
        return Err(ArgsError::NoCommand);
    };
    let rest: Vec<String> = args.collect();
    match command.as_str() {
        "--help" | "-h" | "help" => Ok(Command::Help(Topic::Main)),
        "--version" | "-V" => Ok(Command::Version),
        "new" => parse_new(rest),
        "check" => parse_check(rest),
        "bench" => parse_bench(rest),
        _ => Err(ArgsError::UnknownCommand(command)),
    }
}

/// Sets a boolean flag once.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the parser hands every option its owned value; a flag must have none"
)]
fn set_flag(slot: &mut bool, name: &'static str, value: Option<String>) -> Result<(), ArgsError> {
    if value.is_some() {
        return Err(ArgsError::UnexpectedValue(name));
    }
    if *slot {
        return Err(ArgsError::Repeated(name));
    }
    *slot = true;
    Ok(())
}

/// Takes an option's value: inline, or the next argument.
fn value(
    name: &'static str,
    inline: Option<String>,
    rest: &mut impl Iterator<Item = String>,
) -> Result<String, ArgsError> {
    inline
        .or_else(|| rest.next())
        .filter(|v| !v.is_empty())
        .ok_or(ArgsError::MissingValue(name))
}

/// Sets an option once.
fn set_once<T>(slot: &mut Option<T>, name: &'static str, v: T) -> Result<(), ArgsError> {
    if slot.is_some() {
        return Err(ArgsError::Repeated(name));
    }
    *slot = Some(v);
    Ok(())
}

fn parse_new(rest: Vec<String>) -> Result<Command, ArgsError> {
    let mut it = rest.into_iter();
    let (mut name, mut path, mut id, mut agents, mut turns) = (None, None, None, None, None);
    let (mut snapshots, mut tessera_path) = (false, None);
    while let Some(arg) = it.next() {
        match split(arg) {
            Arg::Flag(flag, inline) => match flag.as_str() {
                "--help" => return Ok(Command::Help(Topic::New)),
                "--path" => {
                    let v = value("--path", inline, &mut it)?;
                    set_once(&mut path, "--path", PathBuf::from(v))?;
                }
                "--id" => {
                    let v = value("--id", inline, &mut it)?;
                    set_once(&mut id, "--id", v)?;
                }
                "--agents" => {
                    let v = value("--agents", inline, &mut it)?;
                    let n = v
                        .parse::<u16>()
                        .ok()
                        .filter(|n| (1..=MAX_AGENTS).contains(n))
                        .ok_or(ArgsError::InvalidValue {
                            option: "--agents",
                            value: v,
                            reason: "an integer in 1..=256",
                        })?;
                    set_once(&mut agents, "--agents", n)?;
                }
                "--turns" => {
                    let v = value("--turns", inline, &mut it)?;
                    let t = match v.as_str() {
                        "simultaneous" => Turns::Simultaneous,
                        "sequential" => Turns::Sequential,
                        "mixed" => Turns::Mixed,
                        _ => {
                            return Err(ArgsError::InvalidValue {
                                option: "--turns",
                                value: v,
                                reason: "simultaneous, sequential or mixed",
                            });
                        }
                    };
                    set_once(&mut turns, "--turns", t)?;
                }
                "--snapshots" => set_flag(&mut snapshots, "--snapshots", inline)?,
                "--tessera-path" => {
                    let v = value("--tessera-path", inline, &mut it)?;
                    set_once(&mut tessera_path, "--tessera-path", PathBuf::from(v))?;
                }
                _ => {
                    return Err(ArgsError::UnknownOption {
                        command: "new",
                        option: flag,
                    });
                }
            },
            Arg::Positional(p) => {
                if name.is_some() {
                    return Err(ArgsError::Unexpected(p));
                }
                name = Some(p);
            }
        }
    }
    Ok(Command::New(NewArgs {
        name: name.ok_or(ArgsError::MissingName)?,
        path,
        id,
        agents: agents.unwrap_or(1),
        turns: turns.unwrap_or(Turns::Simultaneous),
        snapshots,
        tessera_path,
    }))
}

/// Parses `--locked` and `--offline`; `false` for any other flag.
fn cargo_flag(
    cargo: &mut CargoFlags,
    flag: &str,
    inline: Option<String>,
) -> Result<bool, ArgsError> {
    match flag {
        "--locked" => set_flag(&mut cargo.locked, "--locked", inline).map(|()| true),
        "--offline" => set_flag(&mut cargo.offline, "--offline", inline).map(|()| true),
        _ => Ok(false),
    }
}

fn parse_check(rest: Vec<String>) -> Result<Command, ArgsError> {
    let mut out = CheckArgs::default();
    for arg in rest {
        match split(arg) {
            Arg::Flag(flag, inline) => match flag.as_str() {
                "--help" => return Ok(Command::Help(Topic::Check)),
                "--python" => set_flag(&mut out.python, "--python", inline)?,
                "--bless" => set_flag(&mut out.bless, "--bless", inline)?,
                "--uv" => set_flag(&mut out.uv, "--uv", inline)?,
                _ => {
                    if !cargo_flag(&mut out.cargo, &flag, inline)? {
                        return Err(ArgsError::UnknownOption {
                            command: "check",
                            option: flag,
                        });
                    }
                }
            },
            Arg::Positional(p) => return Err(ArgsError::Unexpected(p)),
        }
    }
    if out.uv && !out.python {
        return Err(ArgsError::InvalidValue {
            option: "--uv",
            value: "--uv".to_owned(),
            reason: "only with --python",
        });
    }
    Ok(Command::Check(out))
}

fn parse_bench(rest: Vec<String>) -> Result<Command, ArgsError> {
    let mut out = BenchArgs::default();
    for arg in rest {
        match split(arg) {
            Arg::Flag(flag, inline) => match flag.as_str() {
                "--help" => return Ok(Command::Help(Topic::Bench)),
                "--quick" => set_flag(&mut out.quick, "--quick", inline)?,
                _ => {
                    if !cargo_flag(&mut out.cargo, &flag, inline)? {
                        return Err(ArgsError::UnknownOption {
                            command: "bench",
                            option: flag,
                        });
                    }
                }
            },
            Arg::Positional(p) => return Err(ArgsError::Unexpected(p)),
        }
    }
    Ok(Command::Bench(out))
}

/// The help text of `topic`.
#[must_use]
pub const fn help(topic: Topic) -> &'static str {
    match topic {
        Topic::Main => {
            "cargo tessera: scaffold, check and bench Tessera host environment crates

Usage: cargo tessera <command> [options]

Commands:
  new <name>   Create a host environment crate (its own workspace and wheel)
  check        Run the determinism, continuation, space, invariant and
               golden checks; with --python, the adapter tests too
  bench        Run the crate's criterion bench and print a summary

Options:
  -h, --help     Print help (also after a command)
  -V, --version  Print the version
"
        }
        Topic::New => {
            "Usage: cargo tessera new <name> [options]

Creates the directory <name> (or --path) with a host environment crate:
Cargo.toml, src/lib.rs, tests/check.rs, tests/test_adapters.py,
benches/step.rs, pyproject.toml and its Python package. Refuses an
existing directory.

Options:
  --path <dir>            Directory to create [default: ./<name>]
  --id <ns/Name-vN>       Environment id; `tessera/` is reserved
                          [default: <name>/<Name>-v1]
  --agents <n>            Agent slots, 1..=256 [default: 1]
  --turns <turns>         simultaneous, sequential or mixed
                          [default: simultaneous]
  --snapshots             Implement HostSnapshot and check it
  --tessera-path <dir>    Depend on a Tessera checkout by path instead of
                          the exact version of this tool
"
        }
        Topic::Check => {
            "Usage: cargo tessera check [options]

Run in a scaffolded crate. Runs `cargo test --test check`: determinism
(Inline and with workers), continuation, spaces, invariants, snapshots
(if implemented) and the golden hashes in tests/golden.json.

Options:
  --bless      Record tests/golden.json instead of comparing it
  --python     Also install the wheel into a virtual environment (the
               active one if VIRTUAL_ENV is set, else a new one under
               target/) and run pytest tests/test_adapters.py
  --uv         With --python: install with `uv pip install` instead of
               `maturin develop` (both must be on PATH)
  --locked     Pass --locked to cargo (and to the wheel build)
  --offline    Pass --offline to cargo (and to the wheel build)
"
        }
        Topic::Bench => {
            "Usage: cargo tessera bench [options]

Run in a scaffolded crate. Runs `cargo bench --bench step` and prints steps
per second per pool size and mode, and snapshot costs if implemented, from
criterion's estimates. The numbers hold for this machine only.

Options:
  --quick      Criterion's --quick: a short run
  --locked     Pass --locked to cargo
  --offline    Pass --offline to cargo
"
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        ArgsError, BenchArgs, CargoFlags, CheckArgs, Command, NewArgs, Topic, Turns, parse,
    };

    fn argv(args: &[&str]) -> Vec<String> {
        std::iter::once("cargo-tessera")
            .chain(args.iter().copied())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn test_parse_new_defaults_are_one_simultaneous_agent() {
        let got = parse(argv(&["tessera", "new", "my-env"])).unwrap();
        assert_eq!(
            got,
            Command::New(NewArgs {
                name: "my-env".to_owned(),
                path: None,
                id: None,
                agents: 1,
                turns: Turns::Simultaneous,
                snapshots: false,
                tessera_path: None,
            })
        );
    }

    #[test]
    fn test_parse_new_every_option_both_spellings_parsed() {
        let got = parse(argv(&[
            "new",
            "--path=out/dir",
            "my-env",
            "--id",
            "acme/Grid-v2",
            "--agents=3",
            "--turns",
            "sequential",
            "--snapshots",
            "--tessera-path",
            "/repo",
        ]))
        .unwrap();
        assert_eq!(
            got,
            Command::New(NewArgs {
                name: "my-env".to_owned(),
                path: Some(PathBuf::from("out/dir")),
                id: Some("acme/Grid-v2".to_owned()),
                agents: 3,
                turns: Turns::Sequential,
                snapshots: true,
                tessera_path: Some(PathBuf::from("/repo")),
            })
        );
    }

    #[test]
    fn test_parse_check_and_bench_flags_parsed() {
        assert_eq!(
            parse(argv(&[
                "tessera",
                "check",
                "--python",
                "--bless",
                "--uv",
                "--locked",
                "--offline"
            ]))
            .unwrap(),
            Command::Check(CheckArgs {
                python: true,
                bless: true,
                uv: true,
                cargo: CargoFlags {
                    locked: true,
                    offline: true,
                },
            })
        );
        assert_eq!(
            parse(argv(&["bench", "--quick", "--offline"])).unwrap(),
            Command::Bench(BenchArgs {
                quick: true,
                cargo: CargoFlags {
                    locked: false,
                    offline: true,
                },
            })
        );
    }

    #[test]
    fn test_parse_help_per_subcommand_returns_its_topic() {
        assert_eq!(
            parse(argv(&["--help"])).unwrap(),
            Command::Help(Topic::Main)
        );
        assert_eq!(
            parse(argv(&["tessera", "help"])).unwrap(),
            Command::Help(Topic::Main)
        );
        assert_eq!(
            parse(argv(&["new", "-h"])).unwrap(),
            Command::Help(Topic::New)
        );
        assert_eq!(
            parse(argv(&["check", "--help"])).unwrap(),
            Command::Help(Topic::Check)
        );
        assert_eq!(
            parse(argv(&["bench", "--help"])).unwrap(),
            Command::Help(Topic::Bench)
        );
        assert_eq!(parse(argv(&["--version"])).unwrap(), Command::Version);
    }

    #[test]
    fn test_parse_refusals_each_return_their_error() {
        let cases: &[(&[&str], ArgsError)] = &[
            (&[], ArgsError::NoCommand),
            (&["tessera"], ArgsError::NoCommand),
            (&["build"], ArgsError::UnknownCommand("build".to_owned())),
            (&["new"], ArgsError::MissingName),
            (&["new", "a", "b"], ArgsError::Unexpected("b".to_owned())),
            (&["new", "a", "--path"], ArgsError::MissingValue("--path")),
            (&["new", "a", "--id="], ArgsError::MissingValue("--id")),
            (
                &["new", "a", "--snapshots", "--snapshots"],
                ArgsError::Repeated("--snapshots"),
            ),
            (
                &["new", "a", "--snapshots=yes"],
                ArgsError::UnexpectedValue("--snapshots"),
            ),
            (
                &["new", "a", "--agents", "2", "--agents", "3"],
                ArgsError::Repeated("--agents"),
            ),
            (
                &["new", "a", "--agents", "0"],
                ArgsError::InvalidValue {
                    option: "--agents",
                    value: "0".to_owned(),
                    reason: "an integer in 1..=256",
                },
            ),
            (
                &["new", "a", "--agents", "257"],
                ArgsError::InvalidValue {
                    option: "--agents",
                    value: "257".to_owned(),
                    reason: "an integer in 1..=256",
                },
            ),
            (
                &["new", "a", "--turns", "async"],
                ArgsError::InvalidValue {
                    option: "--turns",
                    value: "async".to_owned(),
                    reason: "simultaneous, sequential or mixed",
                },
            ),
            (
                &["new", "a", "--quick"],
                ArgsError::UnknownOption {
                    command: "new",
                    option: "--quick".to_owned(),
                },
            ),
            (
                &["check", "--quick"],
                ArgsError::UnknownOption {
                    command: "check",
                    option: "--quick".to_owned(),
                },
            ),
            (
                &["check", "extra"],
                ArgsError::Unexpected("extra".to_owned()),
            ),
            (
                &["check", "--locked", "--locked"],
                ArgsError::Repeated("--locked"),
            ),
            (
                &["check", "--uv"],
                ArgsError::InvalidValue {
                    option: "--uv",
                    value: "--uv".to_owned(),
                    reason: "only with --python",
                },
            ),
            (
                &["bench", "--python"],
                ArgsError::UnknownOption {
                    command: "bench",
                    option: "--python".to_owned(),
                },
            ),
            (
                &["bench", "--offline=1"],
                ArgsError::UnexpectedValue("--offline"),
            ),
        ];
        for (args, want) in cases {
            assert_eq!(parse(argv(args)).unwrap_err(), *want, "{args:?}");
        }
    }

    #[test]
    fn test_cargo_flags_args_in_order() {
        let both = CargoFlags {
            locked: true,
            offline: true,
        };
        assert_eq!(both.args(), ["--locked", "--offline"]);
        assert_eq!(CargoFlags::default().args().len(), 0);
    }
}
