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


namespace QSY {

template <class Float>
    requires std::floating_point<Float>
class GateOperationBase {
    using Gate = std::vector<Float>;
  public:
    GateOperationBase(){}
    virtual Gate multiply(const Gate& a, const Gate& b) const = 0;
    virtual Gate dagger(const Gate& a) const = 0;
};

template<class GO, class Float>
concept DerivedFromGateOperationBase = std::is_base_of_v<GateOperationBase<Float>, GO>;



} // namespace QSY