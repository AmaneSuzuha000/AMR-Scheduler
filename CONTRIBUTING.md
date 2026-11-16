# Contributing to AMR-Scheduler

Welcome to **AMR-Scheduler**! We welcome contributions to our multi-robot coordination engine, high-performance reservation kernel, and research extensions.

## 1. Development Setup

### Prerequisites
- Python 3.10+ (tested on 3.11, 3.12, 3.13, 3.14)
- Rust Toolchain 1.75+ (`rustup default stable`)
- .NET 6.0 SDK (optional, for C# RAWSim-O.Live twin)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/AmaneSuzuha000/AMR-Scheduler.git
cd AMR-Scheduler

# Build the native Rust acceleration kernel
cargo build --release --manifest-path crates/Cargo.toml

# Install Python package in editable mode with development dependencies
python -m pip install -e ".[dev,viz,web]"
```

---

## 2. Running Tests

### Python Test Suite
```bash
# Run all unit and integration tests
python -m pytest -q

# Run CVCR differential and closed-loop tests
python -m pytest Tests/test_cvcr_*.py -v

# Run docstring integrity verification
python -m pytest Tests/test_docstring_integrity.py
```

### Rust Kernel Tests
```bash
cargo test --manifest-path crates/Cargo.toml
```

### End-to-End Chain Verification
```bash
python tools/verify_all_chain.py --fast
```

---

## 3. Code Standards & Architecture Guidelines

1. **Docstring Integrity**:
   Every public module, class, method, and function must include descriptive docstrings explaining intent, arguments, and return types (enforced by `test_docstring_integrity.py`).
2. **Rust FFI Boundaries**:
   All new C-ABI functions in `crates/*/src/ffi.rs` must be guarded with `ffi_guard!` (`std::panic::catch_unwind`) to prevent internal panics from propagating across the boundary into Python or C#.
3. **Commitment & Safety Invariants**:
   When implementing rescheduling or allocation policies, never violate irrevocable task commitments (tasks already picked or in transit). Always validate modifications through `NativeTransactionOwner` and audit trajectories with `IndependentAuditor`.
