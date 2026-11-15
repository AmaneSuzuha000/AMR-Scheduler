// Copyright 2026 AMR project. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cmath>

namespace amr_execution
{
// Command-watchdog limits. Values are seconds or metres per second. The guard
// plugin reads the same names from its SDF block and falls back to these
// defaults, so the contract stays readable without a rebuilt world.
struct WatchdogLimits
{
  double command_timeout;          // silence tolerated while motion is possible
  double lease_timeout;            // mission lease must be refreshed this often
  double initial_command_timeout;  // allowance before an action's first command
  double park_speed;               // coarse planar speed band for a parked body
  double park_wz;                  // coarse |yaw rate| band for a parked body
  double park_confirm;             // parked evidence before the grace applies
  double park_grace;               // bounded silence allowance while parked
};

// A zero target on a body that is no longer meaningfully moving cannot be moved
// by the guard, so a controller that falls silent only after it has commanded
// zero is not an uncontrolled-motion hazard. The band is deliberately coarse:
// this is not the strict stop proof used before a new permit, only evidence
// that nothing is being driven. The allowance is bounded: silence beyond
// park_confirm + park_grace is still a fault, and a nonzero target, or a body
// that is still moving or turning, keeps the ordinary 0.25 s command watchdog
// in force unchanged.
inline bool ParkedTarget(const WatchdogLimits & limits, double target_linear,
  double target_angular, double actual_speed, double actual_wz)
{
  return std::abs(target_linear) < 1e-12 && std::abs(target_angular) < 1e-12 &&
         actual_speed < limits.park_speed && std::abs(actual_wz) < limits.park_wz;
}

struct WatchdogInput
{
  bool armed;
  bool have_command;
  bool have_lease;
  double command_age;
  double lease_age;
  double park_held_s;
};

struct WatchdogDecision
{
  bool latch;
  const char * reason;
};

inline WatchdogDecision EvaluateWatchdog(const WatchdogLimits & limits, const WatchdogInput & input)
{
  if (!input.armed) {return {false, ""};}
  if (input.lease_age > limits.lease_timeout) {return {true, "mission lease timeout"};}
  if (!input.have_command) {
    // The first command of an action keeps its own allowance. Park evidence
    // cannot excuse an action that never started moving.
    return {input.command_age > limits.initial_command_timeout,
            "velocity command timeout"};
  }
  const bool parked = input.have_lease && input.park_held_s >= limits.park_confirm &&
    input.park_held_s <= limits.park_confirm + limits.park_grace;
  if (input.command_age > limits.command_timeout && !parked) {
    return {true, "velocity command timeout"};
  }
  return {false, ""};
}
}  // namespace amr_execution
