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
#include <fstream>
#include <cassert>
#include <string>
#include <optional>
#include <array>
#include <queue>
#include <utility>
#include <filesystem>

#include <hnswlib/hnswlib.h>

#include "utils.hpp"

// HNSWlib
// https://github.com/nmslib/hnswlib


// ref. https://arxiv.org/abs/1310.6813
constexpr int count_Clifford(int Q) {
    int c = 1;
    for(int i = 1; i <= Q; i++) {
        c *= (1<<(i*2))-1;
        c <<= i*2+1;
    }
    return c;
}
constexpr int const_log2(int D) {
    int q = 0;
    for(;(1<<q)<D;q++);
    return q;
}

template <int D>
class PhaseInvariantCliffordSpace : public hnswlib::SpaceInterface<float> {
    size_t _dim;
    size_t _data_size;
    hnswlib::DISTFUNC<float> _dist_func;
    QSY::PhaseInvariantSpace<float, D> _another_space;
    std::optional<hnswlib::HierarchicalNSW<float>> _alg_hnsw;
    float* _Cliffords;
    int _M, _ef_construction;
  public:
    // This method is prone to loss of significance.
    // If that happens (a typical symptom is the error becoming 0), it might be better to multiply by a D-th root of unity for the global phase and rely on the L2 norm instead.
    PhaseInvariantCliffordSpace(int dim) : _dim(dim), _data_size(dim*sizeof(float)), _M(16), _ef_construction(200), _another_space(dim) {
        std::string Clifford_path = std::string("data/Clifford") + std::to_string(const_log2(D)) + std::string(".fvecs");
        unsigned num, dim__;
        load_fvecs(Clifford_path.c_str(), _Cliffords, num, dim__);
        assert(num == count_Clifford(const_log2(D)));
        assert(dim__ == unsigned(dim));

        const std::string hnsw_path = std::string("data/hnsw_Clifford") + std::to_string(const_log2(D)) + std::string(".bin");
        if(!std::filesystem::exists(hnsw_path)) {
            _alg_hnsw.emplace(&_another_space, count_Clifford(const_log2(D)), _M, _ef_construction);
            for(int i = 0; i < num; i++) {
                _alg_hnsw->addPoint(_Cliffords + i*dim, i);
            }
            _alg_hnsw->saveIndex(hnsw_path);
        }
        else {
            _alg_hnsw.emplace(&_another_space, hnsw_path);
        }

        _dist_func = [](const void* p1, const void* p2, const void* dim_ptr) -> float {
            auto* self = reinterpret_cast<const PhaseInvariantCliffordSpace*>(dim_ptr);
            const float* a = (const float*)p1;
            const float* b = (const float*)p2;
            std::array<float, D*D*2> a_bdg{};
            a_bdg.fill(0.f);
            for(int i = 0; i < D; i++) {
                for(int j = 0; j < D; j++) {
                    for(int k = 0; k < D; k++) {
                        a_bdg[(i*D+k)*2] += a[(i*D+j)*2]*b[(k*D+j)*2] + a[(i*D+j)*2+1]*b[(k*D+j)*2+1];
                        a_bdg[(i*D+k)*2+1] += -a[(i*D+j)*2]*b[(k*D+j)*2+1] + a[(i*D+j)*2+1]*b[(k*D+j)*2];
                    }
                }
            }
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = self->_alg_hnsw->searchKnn(a_bdg.data(), 1);
            return result.top().first;
        };
    }
    size_t get_data_size() override { return _data_size; }
    hnswlib::DISTFUNC<float> get_dist_func() override { return _dist_func; }
    void* get_dist_func_param() override { return this; }
};