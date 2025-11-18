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

#include <mitmnns/gate_operation.hpp>


namespace QSY {
template <class F, int D>
class SUD : GateOperationBase<F> {
    using Gate = std::vector<F>;
  public:
    Gate multiply(const Gate& a, const Gate& b) const override {
        Gate res(D*D*2);
        for(int i = 0; i < D; i++) {
            for(int j = 0; j < D; j++) {
                for(int k = 0; k < D; k++) {
                    res[(i*D+k)*2] += a[(i*D+j)*2]*b[(j*D+k)*2] - a[(i*D+j)*2+1]*b[(j*D+k)*2+1];
                    res[(i*D+k)*2+1] += a[(i*D+j)*2]*b[(j*D+k)*2+1] + a[(i*D+j)*2+1]*b[(j*D+k)*2];
                }
            }
        }
        return res;
    }
    Gate dagger(const Gate& a) const override {
        Gate res(D*D*2);
        for(int i = 0; i < D; i++) {
            for(int j = 0; j < D; j++) {
                res[(i*D+j)*2] = a[(j*D+i)*2];
                res[(i*D+j)*2+1] = -a[(j*D+i)*2+1];
            }
        }
        return res;
    }
    
    bool is_U(const Gate& a, const F eps=1e-5) const {
        Gate ad = dagger(a);
        Gate i_hatena = multiply(a, ad);
        F l1 = 0.;
        for(int i = 0; i < D; i++) {
            for(int j = 0; j < D; j++) {
                if(i == j) l1 += abs(i_hatena[i*(D+1)*2]-F(1.)) + abs(i_hatena[i*(D+1)*2+1]);
                else l1 += abs(i_hatena[(i*D+j)*2]) + abs(i_hatena[(i*D+j)*2+1]);
            }
        }
        return l1 < eps;
    }
};

} // namespace QSY