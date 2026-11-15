//! Discrete whole-step half-open intervals `[start, end)`.
//!
//! # Semantics & Rationale
//! As detailed in the historical system architecture, continuous floating-point
//! intervals with absolute tolerances (such as `1e-12`) inevitably fail when simulation
//! times grow (e.g. `start + TOLERANCE == start` due to double ULP expansion).
//!
//! Whole-step integer intervals completely eradicate the ULP collapse at the root.
//! Adjacency is clean: `[0, 5)` and `[5, 10)` do not overlap.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntervalError {
    InvalidInterval { start: i64, end: i64 },
}

impl fmt::Display for IntervalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInterval { start, end } => {
                write!(f, "invalid interval [{}, {})", start, end)
            }
        }
    }
}

impl std::error::Error for IntervalError {}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Interval {
    pub start: i64,
    pub end: i64,
}

impl fmt::Debug for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

impl Interval {
    /// Creates a new half-open interval `[start, end)`.
    ///
    /// # Panics
    /// Panics if `end < start`. For a fallible version, use [`Interval::try_new`].
    #[inline]
    pub fn new(start: i64, end: i64) -> Self {
        Self::try_new(start, end).unwrap_or_else(|e| panic!("{}", e))
    }

    /// Creates a new half-open interval `[start, end)`.
    #[inline]
    pub fn try_new(start: i64, end: i64) -> Result<Self, IntervalError> {
        if end < start {
            Err(IntervalError::InvalidInterval { start, end })
        } else {
            Ok(Self { start, end })
        }
    }

    /// Creates an unchecked interval (skips validation in performance-critical inner loops).
    #[inline(always)]
    pub const fn new_unchecked(start: i64, end: i64) -> Self {
        Self { start, end }
    }

    /// Length of the interval in discrete whole steps.
    #[inline(always)]
    pub const fn len(&self) -> i64 {
        self.end - self.start
    }

    /// Whether this interval is empty (`start == end`).
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Returns `true` if this interval overlaps with `other`.
    ///
    /// Two half-open intervals `[s1, e1)` and `[s2, e2)` overlap if and only if:
    /// `s1 < e2 && s2 < e1`.
    #[inline(always)]
    pub const fn overlaps(&self, other: &Interval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Returns `true` if `step` is within `[start, end)`.
    #[inline(always)]
    pub const fn contains_step(&self, step: i64) -> bool {
        self.start <= step && step < self.end
    }

    /// Computes the intersection of two intervals, returning `None` if they do not overlap.
    #[inline]
    pub fn intersection(&self, other: &Interval) -> Option<Interval> {
        let s = self.start.max(other.start);
        let e = self.end.min(other.end);
        if s < e {
            Some(Interval::new_unchecked(s, e))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_basic() {
        let iv = Interval::new(5, 10);
        assert_eq!(iv.len(), 5);
        assert!(!iv.is_empty());
        assert_eq!(format!("{}", iv), "[5, 10)");
    }

    #[test]
    fn test_interval_empty() {
        let iv = Interval::new(5, 5);
        assert_eq!(iv.len(), 0);
        assert!(iv.is_empty());
    }

    #[test]
    fn test_interval_invalid() {
        assert!(Interval::try_new(10, 5).is_err());
    }

    #[test]
    fn test_interval_overlap() {
        let a = Interval::new(10, 20);
        let b = Interval::new(15, 25);
        let c = Interval::new(20, 30);
        let d = Interval::new(0, 10);

        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
        // Half-open: adjacent intervals [10, 20) and [20, 30) do NOT overlap
        assert!(!a.overlaps(&c));
        assert!(!c.overlaps(&a));
        assert!(!a.overlaps(&d));
    }

    #[test]
    fn test_interval_contains_and_intersection() {
        let a = Interval::new(10, 20);
        assert!(a.contains_step(10));
        assert!(a.contains_step(19));
        assert!(!a.contains_step(9));
        assert!(!a.contains_step(20));

        let b = Interval::new(15, 25);
        let inter = a.intersection(&b);
        assert_eq!(inter, Some(Interval::new(15, 20)));

        let c = Interval::new(20, 30);
        assert_eq!(a.intersection(&c), None);
    }
}
