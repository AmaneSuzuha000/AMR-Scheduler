// Copyright 2026 AMR project. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace amr_execution
{
struct BoundedTwist {double linear; double angular; double scale;};

// A common nonnegative scale preserves curvature, sign and the collision-checked arc.
// This is a controller output constraint; the independent guard still rejects bad commands.
inline BoundedTwist BoundTwist(double linear, double angular, double max_linear, double max_angular)
{
  if (!std::isfinite(linear) || !std::isfinite(angular) ||
    !std::isfinite(max_linear) || !std::isfinite(max_angular) ||
    max_linear <= 0. || max_angular <= 0.)
  {
    throw std::invalid_argument("twist and positive limits must be finite");
  }
  double scale = 1.;
  if (std::abs(linear) > max_linear) {scale = std::min(scale, max_linear / std::abs(linear));}
  if (std::abs(angular) > max_angular) {scale = std::min(scale, max_angular / std::abs(angular));}
  return {linear * scale, angular * scale, scale};
}
}  // namespace amr_execution
