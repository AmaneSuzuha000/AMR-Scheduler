// Copyright 2026 AMR project. SPDX-License-Identifier: Apache-2.0
#include "../src/twist_bounds.hpp"
#include <iostream>
#include <limits>

int main()
{
  using amr_execution::BoundTwist;
  int checks = 0;
  for (const double linear : {-2., -.2, -.05, 0., .05, .2, 2.}) {
    for (const double angular : {-10., -1.1884979822651958, -.8, 0., .8, 1.2, 10.}) {
      const auto bounded = BoundTwist(linear, angular, .2, .8);
      if (std::abs(bounded.linear) > .2 + 1e-15 || std::abs(bounded.angular) > .8 + 1e-15 ||
        bounded.scale <= 0. || bounded.scale > 1. ||
        std::abs(bounded.linear * angular - bounded.angular * linear) > 1e-13 ||
        (std::abs(linear) <= .2 && std::abs(angular) <= .8 && bounded.scale != 1.)) {return 1;}
      ++checks;
    }
  }
  for (double invalid : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {BoundTwist(.1, .1, invalid, .8);} catch (const std::invalid_argument &) {rejected = true;}
    if (!rejected) {return 2;}
    ++checks;
  }
  for (double invalid : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {BoundTwist(.1, invalid, .2, .8);} catch (const std::invalid_argument &) {rejected = true;}
    if (!rejected) {return 3;}
    ++checks;
  }
  const auto historical = BoundTwist(.05, -1.1884979822651958, .2, .8);
  if (historical.angular < -.800000000000001 || historical.scale >= 1. || historical.linear <= 0.) {return 4;}
  std::cout << "{\"passed\":true,\"cases\":" << checks + 1 <<
    ",\"historical_raw_angular\":-1.1884979822651958,\"bounded_angular\":" << historical.angular << "}\n";
  return 0;
}
