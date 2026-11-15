// Copyright 2026 AMR project. SPDX-License-Identifier: Apache-2.0
#include "../src/motion_watchdog.hpp"
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
using amr_execution::EvaluateWatchdog;
using amr_execution::ParkedTarget;
using amr_execution::WatchdogDecision;
using amr_execution::WatchdogInput;
using amr_execution::WatchdogLimits;

WatchdogLimits Limits()
{
  return WatchdogLimits{.25, .5, 1., .05, .1, .05, 1.5};
}

int checks = 0;

bool Expect(bool condition, const std::string & label, int code)
{
  ++checks;
  if (!condition) {
    std::cerr << "failed: " << label << "\n";
    std::exit(code);
  }
  return true;
}
}  // namespace

int main()
{
  const auto limits = Limits();

  // Ordinary motion: the .25 s watchdog is unchanged and still bites.
  WatchdogInput moving{true, true, true, .24, .05, 0.};
  Expect(!EvaluateWatchdog(limits, moving).latch, "moving 0.24 s is tolerated", 1);
  moving.command_age = .26;
  const WatchdogDecision late = EvaluateWatchdog(limits, moving);
  Expect(late.latch && std::string(late.reason) == "velocity command timeout",
         "moving 0.26 s latches a velocity timeout", 2);

  // A parked body may not excuse silence before the evidence is confirmed.
  WatchdogInput parked{true, true, true, .26, .05, 0.};
  parked.park_held_s = .04;
  Expect(EvaluateWatchdog(limits, parked).latch,
         "park evidence below park_confirm still latches", 3);
  parked.park_held_s = .05;
  Expect(!EvaluateWatchdog(limits, parked).latch,
         "confirmed park tolerates the goal-completion handoff", 4);
  parked.park_held_s = 1.4;
  Expect(!EvaluateWatchdog(limits, parked).latch, "park grace covers the bounded handoff", 5);
  parked.park_held_s = .05 + 1.5 + 1e-9;
  Expect(EvaluateWatchdog(limits, parked).latch,
         "park grace is bounded; longer silence is a fault", 6);

  // The lease watchdog always outranks the park allowance.
  WatchdogInput stale_lease{true, true, true, .26, .51, .2};
  const WatchdogDecision lease = EvaluateWatchdog(limits, stale_lease);
  Expect(lease.latch && std::string(lease.reason) == "mission lease timeout",
         "lease timeout is never excused by park", 7);

  // An action that never received a command keeps its own allowance.
  WatchdogInput no_command{true, false, true, .99, .05, 0.};
  Expect(!EvaluateWatchdog(limits, no_command).latch, "initial command allowance holds", 8);
  no_command.command_age = 1.01;
  Expect(EvaluateWatchdog(limits, no_command).latch, "initial command allowance expires", 9);
  no_command.command_age = 1.01;
  no_command.park_held_s = .2;
  Expect(EvaluateWatchdog(limits, no_command).latch,
         "park evidence cannot excuse an action that never started", 10);

  // Without a lease there is no park allowance.
  WatchdogInput no_lease{true, true, false, .26, .05, .2};
  Expect(EvaluateWatchdog(limits, no_lease).latch, "park needs a live lease", 11);

  // Locked guard never latches a timeout.
  WatchdogInput locked{false, true, true, 99., 99., 0.};
  Expect(!EvaluateWatchdog(limits, locked).latch, "disabled guard is inert", 12);

  // Park requires both a zero target and a body that is not moving.
  Expect(ParkedTarget(limits, 0., 0., .0, .0), "zero target and still body is parked", 13);
  Expect(!ParkedTarget(limits, .001, 0., .0, .0), "nonzero target is not parked", 14);
  Expect(!ParkedTarget(limits, 0., .001, .0, .0), "nonzero yaw target is not parked", 15);
  Expect(!ParkedTarget(limits, 0., 0., .06, .0), "moving body is not parked", 16);
  Expect(!ParkedTarget(limits, 0., 0., .0, .11), "turning body is not parked", 17);
  Expect(ParkedTarget(limits, 0., 0., .049, .099), "coasting body inside the band is parked", 18);

  // Thresholds are exclusive: exactly at the limit is not yet still.
  Expect(!ParkedTarget(limits, 0., 0., .05, .0), "speed at the band edge is not parked", 19);
  Expect(!ParkedTarget(limits, 0., 0., .0, .1), "yaw rate at the band edge is not parked", 20);
  moving.command_age = .25;
  Expect(!EvaluateWatchdog(limits, moving).latch, "command age at the limit is tolerated", 21);
  stale_lease.lease_age = .5;
  Expect(!EvaluateWatchdog(limits, stale_lease).latch, "lease age at the limit is tolerated", 22);

  std::cout << "{\"passed\":true,\"cases\":" << checks <<
    ",\"command_timeout_s\":0.25,\"park_speed\":0.05,\"park_wz\":0.1"
    ",\"park_confirm_s\":0.05,\"park_grace_s\":1.5}\n";
  return 0;
}
