[build-system]
requires = ["maturin>=1.16,<2"]
build-backend = "maturin"

[project]
name = "{{crate_name}}"
version = "0.1.0"
description = "A Tessera host environment: {{env_id}}"
requires-python = ">=3.11"
# The host object needs the first-party package (SPEC 14.4.1).
dependencies = ["{{py_tessera_req}}"]

[tool.maturin]
python-source = "python"
module-name = "{{py_package}}._tessera_host"
# The Cargo feature that builds the extension module.
features = ["python"]

# Host protocol 1 discovery (SPEC 14.4.1): tessera_rl.make finds the
# environment through this entry point.
[project.entry-points."tessera_rl.envs"]
"{{env_id}}" = "{{py_package}}._tessera_host:make"
