# ADR 001: Selection of C-ABI cdylib + ctypes over PyO3 for Native Acceleration

## Context
AMR-Scheduler requires high-throughput spatiotemporal reservation collision detection and A*/SIPP path planning to evaluate dozens of candidate schedules within a $150\,\text{ms}$ online decision budget.

Two primary integration options were evaluated:
1. **PyO3 Extension Modules (`cpython` bindings)**: Compiles Rust into a Python C-extension (`.so` / `.pyd`) compiled against specific Python C-API headers.
2. **Standard C-ABI `cdylib` loaded via `ctypes` / P/Invoke**: Compiles Rust into a standard shared library exposing plain `extern "C"` functions with primitive types and pointers.

## Decision
We chose **Standard C-ABI `cdylib` via `ctypes`** for the core native components (`reservation_core` and `spacetime_planner`).

## Consequences & Trade-Offs

### Advantages
1. **Multi-Host Sharing (Python & C#)**:
   The same compiled `.so` / `.dll` artifact serves both Python (`amr_sched`) and C# (`RAWSimO.Live`), eliminating code duplication and binary fragmentation.
2. **Broad Python Interpreter Compatibility**:
   C-ABI shared libraries do not link against Python internal symbols (`libpython.so`). The same dynamic library runs without recompilation across Python 3.10, 3.11, 3.12, 3.13, 3.14, and PyPy.
3. **Build Pipeline Decoupling**:
   `pip install -e .` does not require `setuptools-rust` or a C-compiler toolchain if the precompiled cdylib is placed in the release path or located via `AMR_RESERVATION_CORE_PATH`.
4. **Panic Isolation**:
   By wrapping FFI entry points with `std::panic::catch_unwind`, Rust panics cannot unwind into external runtimes, preventing fatal `SIGABRT` crashes.

### Mitigations for C-ABI Limitations
- **Manual Data Marshalling**: CSV strings and primitive arrays are passed across FFI rather than high-level Python dicts. In our profiling, serializing 10–50 waypoint hops adds $< 0.05\,\text{ms}$, well within our performance budget.
- **Pointer Lifetime Safety**: Every heap-allocated Rust structure (`ReservationTable`, `PlannerGraph`, `ReservationTransaction`) provides explicit constructor (`_create`) and destructor (`_destroy`) pairs, encapsulated in Python context managers and classes (`__del__` / `close()`).
