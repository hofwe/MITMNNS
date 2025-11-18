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

#include <mitmnns/impl/hnswlib_search.hpp>
#include <mitmnns/impl/SUD_operation.hpp>

#include "utils.hpp"


// HNSWlib
// https://github.com/nmslib/hnswlib



int main() {
    // Puali matrices
    std::vector<float> I(8);
    for(int i = 0; i < 2; i++) {
        I[(i*2+i)*2] = 1.;
    }
    std::vector<float> X(8);
    X[3] = X[5] = 1.;
    std::vector<float> Y(8);
    Y[2] = 1.;
    Y[4] = -1.;
    std::vector<float> Z(8);
    Z[1] = 1.;
    Z[7] = -1.;
    std::vector<std::vector<float>> Paulis({I, X, Y, Z});
    std::vector<std::vector<float>> Paulis_2(16);
    for(int i = 0; i < 16; i++) {
        Paulis_2[i] = tensor(Paulis[i>>2], Paulis[i&3]);
    }
    
    
    // prepare T gate net for data1 and data2
    const int max_count = 4;
    const int D = 4;
    T_gen<float, D> t1(std::vector<std::vector<float>>(1, tensor(I, I)), max_count), t2(Paulis_2, max_count);
    std::string prefix = std::string("data/hnsw_T")+std::to_string(D)+"_";

    // single thread
    QSY::hnsw<float, D> hnsw(t2._data, t2._bound, {
                            .hnsw_graph_prefix = prefix,
                            .num_thread_construct = 1,
                            .num_thread_search = 1
                        });
    QSY::MITM<float, QSY::hnsw<float, D>, QSY::SUD<float, D>> mitm(t1._data, t1._bound, &hnsw);

    // unitaries to query
    std::string datafile("data/SU4_100.fvecs");
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
        const float eps = 3e-1;
        std::cout << "Start eps search with eps = " << eps << std::endl;
        for(int i = 0; i < num; i++) {
            std::vector<float> query(random_unitary+i*dim, random_unitary+(i+1)*dim);
            auto [e, id1, id2] = mitm.eps_search(query, eps);

            output(i, e, id1, id2, query, t1, t2);
        }
    }

    // count search
    if(do_count_search) {
        const int count = 8;
        std::cout << "Start count search with count = " << count << std::endl;;
        for(int i = 0; i < num; i++) {
            std::vector<float> query(random_unitary+i*dim, random_unitary+(i+1)*dim);
            auto [e, id1, id2] = mitm.count_search(query, count);

            output(i, e, id1, id2, query, t1, t2);
        }
    }

    // count k search
    if(do_count_k_search) {
        const int k = 10;
        const int count = 8;
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