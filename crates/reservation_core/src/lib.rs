//! `reservation_core`: Fast spatiotemporal reservation table and interval interlocking core for AMR-Scheduler.

pub mod ffi;
pub mod interval;
pub mod table;
pub mod transaction;

pub use interval::{Interval, IntervalError};
pub use table::{ReservationEntry, ReservationTable};
pub use transaction::{ExecutionCommitmentLedger, ReservationTransaction, TransactionError, TransactionState};
