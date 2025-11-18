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

#include <vector>
#include <string>
#include <iostream>
#include <tuple>

#include <mitmnns/meet_in_the_middle.hpp>

#include <mitmnns/impl/direct_search.hpp>
#include <mitmnns/impl/SU2_operation.hpp>

#include "utils.hpp"





float L2_dist(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.;
    for(int i = 0; i < int(a.size()); i++) {
        d += (a[i]-b[i]) * (a[i]-b[i]);
    }
    return sqrt(d);
}
std::vector<float> to_quat(const std::vector<float>& su2) {
    assert(int(su2.size()) == 8);
    std::vector<float> quat({(su2[0] + su2[6]) * 0.5f, (su2[3] + su2[5]) * 0.5f, (su2[2] - su2[4]) * 0.5f, (su2[1] - su2[7]) * 0.5f});
    return quat;
}
float SU2_dist(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> _a = to_quat(a), _b = to_quat(b);
    return L2_dist(_a, _b);
}


int main() {
    // Puali matrices
    std::vector<float> I(8);
    for(int i = 0; i < 2; i++) {
        I[(i*2+i)*2] = 1.;
    }
    std::vector<float> mI(8);
    for(int i = 0; i < 2; i++) {
        mI[(i*2+i)*2] = -1.;
    }
    std::vector<float> X(8);
    X[3] = X[5] = 1.;
    std::vector<float> Y(8);
    Y[2] = 1.;
    Y[4] = -1.;
    std::vector<float> Z(8);
    Z[1] = 1.;
    Z[7] = -1.;
    std::vector<std::vector<float>> Paulis({I, mI, X, QSY::SU2<float>().dagger(X), Y, QSY::SU2<float>().dagger(Y), Z, QSY::SU2<float>().dagger(Z)});
    
    
    // prepare V gate net for data1 and data2
    const int max_count = 8;
    const int D = 2;
    V_gen<float, D> v1(std::vector<std::vector<float>>(1, I), max_count), v2(Paulis, max_count);
    QSY::DirectSearch<float> ds(v2._data, v2._bound, SU2_dist);
    QSY::MITM<float, QSY::DirectSearch<float>, QSY::SU2<float>> mitm(v1._data, v1._bound, &ds);

    // unitaries to query
    std::string datafile("data/SU2_100.fvecs");
    float* random_unitary;
    unsigned num, dim;

    load_fvecs(datafile.c_str(), random_unitary, num, dim);

    if(num > 10) num = 10;



    // search
    bool do_eps_search = true;
    bool do_count_search = true;
    bool do_count_k_search = true;

    // eps search
    if(do_eps_search) {
        const float eps = 1e-2;
        std::cout << "Start eps search with eps = " << eps << std::endl;
        for(int i = 0; i < num; i++) {
            std::vector<float> query(random_unitary+i*dim, random_unitary+(i+1)*dim);
            auto [e, id1, id2] = mitm.eps_search(query, eps);

            output(i, e, id1, id2, query, v1, v2);
        }
    }

    // count search
    if(do_count_search) {
        const int count = 10;
        std::cout << "Start count search with count = " << count << std::endl;;
        for(int i = 0; i < num; i++) {
            std::vector<float> query(random_unitary+i*dim, random_unitary+(i+1)*dim);
            auto [e, id1, id2] = mitm.count_search(query, count);

            output(i, e, id1, id2, query, v1, v2);
        }
    }

    // count k search
    if(do_count_k_search) {
        const int k = 10;
        const int count = 10;
        std::cout << "Start count k search with k = " << k << ", count = " << count << std::endl;
        for(int i = 0; i < num; i++) {
            std::vector<float> query(random_unitary+i*dim, random_unitary+(i+1)*dim);
            auto result = mitm.count_k_search(query, count, k);

            std::cout << i << std::endl;
            std::cout << "error: ";
            for(int i = 0; i < k; i++) std::cout << std::get<0>(result[i]) << ' ';
            std::cout << "\n" << std::endl;
        }
    }
    
    delete[] random_unitary;
}