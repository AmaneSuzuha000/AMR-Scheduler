# Contributing to AMR-Scheduler

Welcome to **AMR-Scheduler**! We welcome contributions to our multi-robot coordination engine, high-performance Rust reservation kernel, hierarchical CVCR/RL policies, and digital-twin bridges.

## 1. Development Setup

### Prerequisites
- Python `>= 3.10` (CI verified on `3.11` & `3.12`)
- Rust Toolchain `>= 1.75` (`rustup default stable`)
- .NET `6.0` SDK (optional, for `RAWSimO.Live` C# twin and tests)

### Quick Start
```bash
git clone https://github.com/AmaneSuzuha000/AMR-Scheduler.git
cd AMR-Scheduler

# Build the native Rust acceleration kernel
cargo build --release --manifest-path crates/Cargo.toml

# Install Python package in editable mode with dev, visualization, web, and RL dependencies
python -m pip install -e ".[dev,viz,web,rl]"
```

---

## 2. Running Tests

### Python Test Suite
```bash
# Run full suite (shows explicit skip reasons for local-only evidence checks)
python -m pytest -rs

# Run CVCR, V17, and 2026 baseline tests
python -m pytest Tests/test_cvcr_*.py Tests/test_v17_*.py Tests/test_baselines_2026.py -v

# Run AST docstring integrity verification
python -m pytest Tests/test_docstring_integrity.py
```

### Rust & C# Suites
```bash
cargo test --manifest-path crates/Cargo.toml
python -m pytest -q experiments/rust_transition/
dotnet test Tests/RAWSimO.Live.Tests/RAWSimO.Live.Tests.csproj -c Release
```

### End-to-End Chain Verification
```bash
python tools/data_manifest.py --check
python tools/verify_all_chain.py --fast
```

---

## 3. Code Standards & Architecture Guidelines

1. **Docstring Integrity**: Every public module, class, method, and function must include clear docstrings (enforced across every Python file by `Tests/test_docstring_integrity.py`).
2. **Rust FFI Boundaries**: All `extern "C"` functions in `crates/*/src/ffi.rs` must be wrapped with `ffi_guard!` (`std::panic::catch_unwind`) so panics never cross the C-ABI boundary.
3. **Commitment & Safety Invariants**: Never mutate irrevocable commitments (`PICKED`, `LOADING`, `UNLOADING`, or in-edge `_transit`). Route all rescheduling mutations through `NativeTransactionOwner` and verify episodes with `IndependentAuditor`.
4. **Clean Release Hygiene**: Keep inference weights compact in `amr_sched/checkpoints/*.npz`; never commit scratch dumps (`_scratch/`), raw logs, or private host credentials.
