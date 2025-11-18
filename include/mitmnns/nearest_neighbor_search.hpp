/* Copyright 2025 Soichiro Yamazaki. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <concepts>
#include <type_traits>
#include <vector>
#include <tuple>


namespace QSY {

template <class Float>
    requires std::floating_point<Float>
class NearestNeighborSearchBase {
    using Point = std::vector<Float>;
    using Index = int;
  public:
    virtual void construct(int iter_num) = 0;
    virtual std::tuple<Float, Index, Index> query_nearest(const std::vector<Point>& points) const = 0;
    // assume the result is sorted in ascending order
    virtual std::vector<std::tuple<Float, Index, Index>> query_k_nearest(const std::vector<Point>& points, const int k) const = 0;
};

template<class NNS, class Float>
concept DerivedFromNearestNeighborSearchBase = std::is_base_of_v<NearestNeighborSearchBase<Float>, NNS>;



} // namespace QSY