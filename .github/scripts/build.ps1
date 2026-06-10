# Build Script for REACT_RS app

# Run application testing
cargo test --release
python -m pip install pytest
python pytest


# Install maturin & build a Python release
maturin build --release

# Copy output from Rust target dir to wheels dir
mkdir wheels
Copy-Item "./target/wheels/react_rs*.whl" -Destination "./wheels"
