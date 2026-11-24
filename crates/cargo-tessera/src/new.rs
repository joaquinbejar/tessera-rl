// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo tessera new` (#351, SPEC 10.2, ADR-0032): renders the embedded
//! templates into a new directory, never overwriting anything.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::args::{NewArgs, Turns};
use crate::error::ToolError;
use crate::names;
use crate::render::{Vars, render};

/// This tool's version: the exact version scaffolds depend on.
pub const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The templates: the path each renders to (itself a template) and the
/// template's name and text.
const TEMPLATES: &[(&str, &str, &str)] = &[
    (
        "Cargo.toml",
        "Cargo.toml.tpl",
        include_str!("templates/Cargo.toml.tpl"),
    ),
    (
        "src/lib.rs",
        "lib.rs.tpl",
        include_str!("templates/lib.rs.tpl"),
    ),
    (
        "tests/check.rs",
        "check.rs.tpl",
        include_str!("templates/check.rs.tpl"),
    ),
    (
        "tests/test_adapters.py",
        "test_adapters.py.tpl",
        include_str!("templates/test_adapters.py.tpl"),
    ),
    (
        "benches/step.rs",
        "step.rs.tpl",
        include_str!("templates/step.rs.tpl"),
    ),
    (
        "pyproject.toml",
        "pyproject.toml.tpl",
        include_str!("templates/pyproject.toml.tpl"),
    ),
    (
        "python/{{py_package}}/__init__.py",
        "__init__.py.tpl",
        include_str!("templates/__init__.py.tpl"),
    ),
    (
        "README.md",
        "README.md.tpl",
        include_str!("templates/README.md.tpl"),
    ),
    (
        ".gitignore",
        "gitignore.tpl",
        include_str!("templates/gitignore.tpl"),
    ),
];

/// What `new` will write: the crate's directory and each file, rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    /// The directory to create.
    pub root: PathBuf,
    /// Each file, relative to `root`, with its text.
    pub files: Vec<(PathBuf, String)>,
}

/// Validates `args` and renders every template; writes nothing.
///
/// # Errors
///
/// [`ToolError::Name`] for a name or id, [`ToolError::TesseraPath`] for a
/// checkout that is not one, or [`ToolError::Render`] for a template bug.
pub fn plan(args: &NewArgs, cwd: &Path) -> Result<Plan, ToolError> {
    names::check_crate(&args.name)?;
    let package = names::package(&args.name);
    names::check_package(&package)?;
    let id = match &args.id {
        Some(id) => names::check_id(id)?,
        None => names::check_id(&names::default_id(&args.name))?,
    };
    let source = match &args.tessera_path {
        Some(path) => Source::Path(checkout(path, cwd)?),
        None => Source::Version,
    };
    let turns = args.turns;
    let vars = Vars::default()
        .value("crate_name", args.name.clone())
        .value("py_package", package.clone())
        .value("env_id", id.id.clone())
        .value("env_version", id.version.to_string())
        .value("agents", args.agents.to_string())
        .value("turns", turns.name())
        .value("tool_version", TOOL_VERSION)
        .value("dep_core", source.dep("tessera-rl-core", ""))
        .value("dep_macros", source.dep("tessera-rl-macros", ""))
        .value(
            "dep_pyffi",
            source.dep(
                "tessera-rl-pyffi",
                ", features = [\"host\"], optional = true",
            ),
        )
        .value(
            "dep_engine",
            source.dep("tessera-rl-engine", ", features = [\"testing\"]"),
        )
        .value("dep_num", source.dep("tessera-rl-num", ""))
        .value("py_tessera_req", source.python_requirement())
        .flag("snapshots", args.snapshots)
        .flag("path_deps", matches!(source, Source::Path(_)))
        .flag("simultaneous", turns == Turns::Simultaneous)
        .flag("sequential", turns == Turns::Sequential)
        .flag("mixed", turns == Turns::Mixed)
        // The Gymnasium adapter takes one agent acting every step.
        .flag("gym", args.agents == 1 && turns == Turns::Simultaneous)
        // PettingZoo interfaces by turns (ADR-0013, ADR-0065 item 6).
        .flag("parallel", turns != Turns::Sequential)
        .flag("aec", turns != Turns::Mixed);
    let mut files = Vec::with_capacity(TEMPLATES.len());
    for &(dest, name, text) in TEMPLATES {
        let dest = render(name, dest, &vars)?;
        files.push((PathBuf::from(dest), render(name, text, &vars)?));
    }
    let root = args
        .path
        .clone()
        .unwrap_or_else(|| PathBuf::from(&args.name));
    let root = if root.is_absolute() {
        root
    } else {
        cwd.join(root)
    };
    Ok(Plan { root, files })
}

/// Where the Tessera crates come from.
enum Source {
    /// Exactly this tool's version, from the registry.
    Version,
    /// A checkout, as a TOML literal string body (no `'`).
    Path(String),
}

impl Source {
    /// The dependency table of `krate`, with `extra` keys.
    fn dep(&self, krate: &str, extra: &str) -> String {
        match self {
            Self::Version => format!("{{ version = \"={TOOL_VERSION}\"{extra} }}"),
            Self::Path(root) => format!("{{ path = '{root}/crates/{krate}'{extra} }}"),
        }
    }

    /// The `tessera-rl` requirement of the wheel.
    fn python_requirement(&self) -> String {
        match self {
            Self::Version => format!("tessera-rl=={TOOL_VERSION}"),
            Self::Path(_) => "tessera-rl".to_owned(),
        }
    }
}

/// The checkout `path` (relative to `cwd`), checked and made absolute.
fn checkout(path: &Path, cwd: &Path) -> Result<String, ToolError> {
    let fail = |reason| ToolError::TesseraPath {
        path: path.to_path_buf(),
        reason,
    };
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    if !abs.join("crates/tessera-rl-core/Cargo.toml").is_file() {
        return Err(fail(
            "not a Tessera checkout (no crates/tessera-rl-core/Cargo.toml)",
        ));
    }
    let text = abs
        .to_str()
        .ok_or_else(|| fail("not valid UTF-8"))?
        .trim_end_matches(['/', '\\'])
        .to_owned();
    if text.contains('\'') || text.chars().any(char::is_control) {
        return Err(fail("contains a quote or a control character"));
    }
    Ok(text)
}

/// Creates `plan.root` and writes every file; refuses an existing path.
///
/// # Errors
///
/// [`ToolError::Exists`] if the directory or a file exists, or
/// [`ToolError::Io`].
pub fn write(plan: &Plan) -> Result<(), ToolError> {
    if fs::symlink_metadata(&plan.root).is_ok() {
        return Err(ToolError::Exists {
            path: plan.root.clone(),
        });
    }
    let io = |action, path: &Path| {
        let path = path.to_path_buf();
        move |source| ToolError::Io {
            action,
            path,
            source,
        }
    };
    if let Some(parent) = plan.root.parent() {
        fs::create_dir_all(parent).map_err(io("creating", parent))?;
    }
    // `create_dir`, not `create_dir_all`: fails if the directory appeared.
    fs::create_dir(&plan.root).map_err(|source| {
        if source.kind() == std::io::ErrorKind::AlreadyExists {
            ToolError::Exists {
                path: plan.root.clone(),
            }
        } else {
            ToolError::Io {
                action: "creating",
                path: plan.root.clone(),
                source,
            }
        }
    })?;
    for (rel, text) in &plan.files {
        let path = plan.root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(io("creating", parent))?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(io("creating", &path))?;
        file.write_all(text.as_bytes())
            .map_err(io("writing", &path))?;
    }
    Ok(())
}

/// `cargo tessera new`: plans, writes, and reports what it wrote to `out`.
///
/// # Errors
///
/// The errors of [`plan`] and [`write`](fn@write).
pub fn run(args: &NewArgs, cwd: &Path, out: &mut impl Write) -> Result<(), ToolError> {
    let plan = plan(args, cwd)?;
    write(&plan)?;
    let shown = plan.root.display();
    let lines = [
        format!("created {shown}"),
        format!("  then: cd {shown} && cargo tessera check --bless && cargo tessera check"),
        "  with Python: cargo tessera check --python (maturin on the PATH)".to_owned(),
    ];
    for line in lines {
        // Output to a closed stdout is not an error of `new`.
        let _ = writeln!(out, "{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::{TOOL_VERSION, plan, run, write};
    use crate::args::{NewArgs, Turns};
    use crate::error::ToolError;

    static NEXT: AtomicU32 = AtomicU32::new(0);

    /// A path under the temporary directory that does not exist yet.
    fn scratch(tag: &str) -> PathBuf {
        let k = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("cargo-tessera-{tag}-{}-{k}", std::process::id()))
    }

    fn args(name: &str) -> NewArgs {
        NewArgs {
            name: name.to_owned(),
            path: None,
            id: None,
            agents: 1,
            turns: Turns::Simultaneous,
            snapshots: false,
            tessera_path: None,
        }
    }

    /// A fake checkout: just the file `new` looks for.
    fn fake_checkout() -> PathBuf {
        let root = scratch("checkout");
        std::fs::create_dir_all(root.join("crates/tessera-rl-core")).unwrap();
        std::fs::write(root.join("crates/tessera-rl-core/Cargo.toml"), "").unwrap();
        root
    }

    fn file<'a>(plan: &'a super::Plan, rel: &str) -> &'a str {
        plan.files
            .iter()
            .find(|(p, _)| p == Path::new(rel))
            .map(|(_, t)| t.as_str())
            .unwrap()
    }

    #[test]
    fn test_plan_every_variant_renders_without_placeholders() {
        let checkout = fake_checkout();
        for turns in [Turns::Simultaneous, Turns::Sequential, Turns::Mixed] {
            for snapshots in [false, true] {
                for agents in [1, 3] {
                    for path in [None, Some(checkout.clone())] {
                        let mut a = args("my-env");
                        a.turns = turns;
                        a.snapshots = snapshots;
                        a.agents = agents;
                        a.tessera_path = path;
                        let plan = plan(&a, Path::new("/work")).unwrap();
                        assert_eq!(plan.files.len(), 9);
                        for (rel, text) in &plan.files {
                            assert!(!text.contains("{{"), "{rel:?} {turns:?} {snapshots}");
                            assert!(!rel.to_string_lossy().contains("{{"));
                        }
                        let lib = file(&plan, "src/lib.rs");
                        assert_eq!(lib.contains("impl HostSnapshot"), snapshots);
                        assert_eq!(lib.contains("fn acting"), turns != Turns::Simultaneous);
                        assert!(lib.contains("\"my-env/MyEnv-v1\""));
                        let py = file(&plan, "tests/test_adapters.py");
                        assert_eq!(
                            py.contains("check_env"),
                            agents == 1 && turns == Turns::Simultaneous
                        );
                        assert_eq!(py.contains("parallel_api_test"), turns != Turns::Sequential);
                        assert_eq!(
                            py.contains("def test_pettingzoo_aec_api"),
                            turns != Turns::Mixed
                        );
                    }
                }
            }
        }
        std::fs::remove_dir_all(checkout).unwrap();
    }

    #[test]
    fn test_plan_default_pins_exact_tool_version() {
        let plan = plan(&args("grid"), Path::new("/work")).unwrap();
        assert_eq!(plan.root, Path::new("/work/grid"));
        let toml = file(&plan, "Cargo.toml");
        let pin = format!("version = \"={TOOL_VERSION}\"");
        // core, macros, pyffi, engine and num.
        assert_eq!(toml.matches(&pin).count(), 5, "{toml}");
        assert!(!toml.contains("path = "));
        assert!(file(&plan, "pyproject.toml").contains(&format!("tessera-rl=={TOOL_VERSION}")));
        assert!(file(&plan, "python/grid/__init__.py").contains("grid/Grid-v1"));
    }

    #[test]
    fn test_plan_tessera_path_uses_path_dependencies() {
        let checkout = fake_checkout();
        let mut a = args("grid");
        a.tessera_path = Some(checkout.clone());
        let plan = plan(&a, Path::new("/work")).unwrap();
        let toml = file(&plan, "Cargo.toml");
        assert_eq!(toml.matches("path = '").count(), 5, "{toml}");
        assert!(!toml.contains(&format!("\"={TOOL_VERSION}\"")));
        std::fs::remove_dir_all(checkout).unwrap();
    }

    #[test]
    fn test_plan_bad_names_and_paths_refused() {
        assert!(matches!(
            plan(&args("Bad"), Path::new("/w")),
            Err(ToolError::Name(_))
        ));
        let mut a = args("grid");
        a.id = Some("tessera/Grid-v1".to_owned());
        assert!(matches!(plan(&a, Path::new("/w")), Err(ToolError::Name(_))));
        let mut a = args("grid");
        a.tessera_path = Some(scratch("missing"));
        assert!(matches!(
            plan(&a, Path::new("/w")),
            Err(ToolError::TesseraPath { .. })
        ));
    }

    #[test]
    fn test_write_creates_every_file() {
        let root = scratch("write");
        let mut a = args("grid");
        a.path = Some(root.clone());
        a.snapshots = true;
        let mut out = Vec::new();
        run(&a, Path::new("/unused"), &mut out).unwrap();
        for rel in [
            "Cargo.toml",
            "src/lib.rs",
            "tests/check.rs",
            "tests/test_adapters.py",
            "benches/step.rs",
            "pyproject.toml",
            "python/grid/__init__.py",
            "README.md",
            ".gitignore",
        ] {
            assert!(root.join(rel).is_file(), "{rel}");
        }
        assert!(String::from_utf8(out).unwrap().starts_with("created "));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_write_existing_directory_refused_and_untouched() {
        let root = scratch("exists");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("keep.txt"), "mine").unwrap();
        let mut a = args("grid");
        a.path = Some(root.clone());
        let p = plan(&a, Path::new("/unused")).unwrap();
        assert!(matches!(write(&p), Err(ToolError::Exists { .. })));
        let mut out = Vec::new();
        assert!(matches!(
            run(&a, Path::new("/unused"), &mut out),
            Err(ToolError::Exists { .. })
        ));
        let entries: Vec<_> = std::fs::read_dir(&root).unwrap().collect();
        assert_eq!(entries.len(), 1, "nothing written");
        assert_eq!(
            std::fs::read_to_string(root.join("keep.txt")).unwrap(),
            "mine"
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
