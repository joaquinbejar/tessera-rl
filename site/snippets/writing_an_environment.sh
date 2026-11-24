cargo install cargo-tessera
cargo tessera new my-env --id acme/MyEnv-v1 --agents 2 --turns sequential --snapshots
cd my-env
cargo tessera check --bless     # record tests/golden.json, then commit it
cargo tessera check --python    # Rust checks, goldens and the Python adapter tests
cargo tessera bench --quick
