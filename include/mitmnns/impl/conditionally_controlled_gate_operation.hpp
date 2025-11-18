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
#include <vector>
#include <concepts>
#include <cassert>

#include <mitmnns/gate_operation.hpp>


namespace QSY {
template <class F>
class SU2ConditionallyControlledGate : GateOperationBase<F> {
    using Gate = std::vector<F>;
  public:
    Gate multiply(const Gate& a, const Gate& b) const override {
        assert(int(a.size()) == 16);
        assert(int(b.size()) == 8);
        Gate res1(8);
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                for(int k = 0; k < 2; k++) {
                    res1[(i*2+k)*2] += a[(i*2+j)*2]*b[(j*2+k)*2] - a[(i*2+j)*2+1]*b[(j*2+k)*2+1];
                    res1[(i*2+k)*2+1] += a[(i*2+j)*2]*b[(j*2+k)*2+1] + a[(i*2+j)*2+1]*b[(j*2+k)*2];
                }
            }
        }
        Gate res2(8);
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                for(int k = 0; k < 2; k++) {
                    res2[(i*2+k)*2] += res1[(i*2+j)*2]*a[(j*2+k)*2+8] - res1[(i*2+j)*2+1]*a[(j*2+k)*2+9];
                    res2[(i*2+k)*2+1] += res1[(i*2+j)*2]*a[(j*2+k)*2+9] + res1[(i*2+j)*2+1]*a[(j*2+k)*2+8];
                }
            }
        }
        return res2;
    }
    Gate dagger(const Gate& a) const override {
        assert(int(a.size()) == 16);
        Gate res(16);
        for(int i = 0; i < 8; i++) res[i] = a[i];
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                res[(i*2+j)*2+8] = a[(j*2+i)*2+8];
                res[(i*2+j)*2+9] = -a[(j*2+i)*2+9];
            }
        }
        return res;
    }
};

} // namespace QSY