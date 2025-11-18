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
#include <tuple>
#include <optional>
#include <queue>
#include <utility>
#include <string>
#include <filesystem>
#include <cassert>
#include <algorithm>

#include <mitmnns/nearest_neighbor_search.hpp>
#include <hnswlib/hnswlib.h>

#include "hnswlib_utils.h"


// HNSWlib
// https://github.com/nmslib/hnswlib

// for SU(D)
namespace QSY {

template<class SPACE, class F>
concept DerivedFromSpaceInterfaceBase = std::is_base_of_v<hnswlib::SpaceInterface<F>, SPACE>;


template <class F, int D>
class PhaseInvariantSpace : public hnswlib::SpaceInterface<F> {
    size_t _dim;
    size_t _data_size;
    hnswlib::DISTFUNC<F> _dist_func;
    public:
    // This method is prone to loss of significance.
    // If that happens (a typical symptom is the error becoming 0), it might be better to multiply by a D-th root of unity for the global phase and rely on the L2 norm instead.
    PhaseInvariantSpace(int dim) : _dim(dim), _data_size(dim*sizeof(F)) {
        _dist_func = [](const void* p1, const void* p2, const void* dim_ptr) -> F {
            const F* a = (const F*)p1;
            const F* b = (const F*)p2;
            long double real = 0., imag = 0.;
            for(int i = 0; i < D*D*2; i+=2) {
                real += (long double)a[i]*(long double)b[i] + (long double)a[i+1]*(long double)b[i+1];
                imag += (long double)a[i]*(long double)b[i+1] - (long double)a[i+1]*(long double)b[i];
            }
            return F((long double)1. - (real*real + imag*imag)/(D*D))*0.5f;
        };
    }
    size_t get_data_size() override { return _data_size; }
    hnswlib::DISTFUNC<float> get_dist_func() override { return _dist_func; }
    void* get_dist_func_param() override { return &_dim; }
};


template <class F, int D, class SPACE=PhaseInvariantSpace<F, D>>
    requires DerivedFromSpaceInterfaceBase<SPACE, F>
class hnsw : NearestNeighborSearchBase<F> {
    struct Args {
        std::string hnsw_graph_prefix = std::string("");
        int M = 16;
        int ef_construction = 200;
        size_t num_thread_construct = 20;
        size_t num_thread_search = 20;
    };
    std::vector<int> _boundary;
    using Point = std::vector<F>;
    std::vector<Point> _points;
    std::vector<std::optional<hnswlib::HierarchicalNSW<F>>> _alg_hnsw;
    SPACE _space;
    int _now_iter;
    int _M, _ef_construction;
    std::string _hnsw_graph_prefix;
    size_t _num_thread_construct, _num_thread_search;
  public:
    // use designated initializer to provide args
    hnsw(std::vector<Point>& points, std::vector<int>& boundary, const Args& args = {}) : _points(points), _boundary(boundary), _hnsw_graph_prefix(args.hnsw_graph_prefix), _space(D*D*2), _alg_hnsw(int(boundary.size())-1), _M(args.M), _ef_construction(args.ef_construction), _num_thread_construct(args.num_thread_construct), _num_thread_search(args.num_thread_search) {
        assert(_boundary[0] == 0);
        assert(_boundary[int(_boundary.size())-1] == int(points.size()));
    }

    void construct(int iter_num) override {
        assert(iter_num <= int(_boundary.size())-2);
        _now_iter = iter_num;
        if(!_alg_hnsw[iter_num]) {
            std::string hnsw_path = _hnsw_graph_prefix + std::to_string(iter_num) + std::string(".bin");
            if(_hnsw_graph_prefix.empty() or !std::filesystem::exists(hnsw_path)) {
                _alg_hnsw[iter_num].emplace(&_space, _boundary[iter_num+1]-_boundary[iter_num], _M, _ef_construction);
                if(_num_thread_construct > 1) {
                    ParallelFor(_boundary[iter_num], _boundary[iter_num+1], _num_thread_construct, [&](size_t row, size_t threadId) {
                        _alg_hnsw[iter_num]->addPoint(_points[row].data(), row);
                    });
                }
                else {
                    for(int i = _boundary[iter_num]; i < _boundary[iter_num+1]; i++) {
                        _alg_hnsw[iter_num]->addPoint(_points[i].data(), i);
                    }
                }
                if(!_hnsw_graph_prefix.empty()) {
                    _alg_hnsw[iter_num]->saveIndex(hnsw_path);
                }
            }
            else {
                _alg_hnsw[iter_num].emplace(&_space, hnsw_path);
            }
        }
    }

    std::tuple<F, int, int> query_nearest(const std::vector<Point>& points) const override {
        F eps2 = std::numeric_limits<F>::max();
        int ind = -1;
        int nind = -1;
        if(_num_thread_search > 1) {
            std::vector<std::pair<F, hnswlib::labeltype>> results(int(points.size()));
            ParallelFor(0, int(points.size()), _num_thread_search, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<F, hnswlib::labeltype>> result = _alg_hnsw[_now_iter]->searchKnn(points[row].data(), 1);
                results[row] = result.top();
            });
            for(int i = 0; i < int(points.size()); i++) {
                auto [now_eps, now_nind] = results[i];
                if(now_eps < eps2) {
                    eps2 = now_eps;
                    ind = i;
                    nind = int(now_nind);
                }
            }
        }
        else {
            for(int i = 0; i < int(points.size()); i++) {
                std::priority_queue<std::pair<F, hnswlib::labeltype>> result = _alg_hnsw[_now_iter]->searchKnn(points[i].data(), 1);
                auto [now_eps, now_nind] = result.top();
                if(now_eps < eps2) {
                    eps2 = now_eps;
                    ind = i;
                    nind = int(now_nind);
                }
            }
        }
        return {sqrt(eps2), ind, nind};
    }

    std::vector<std::tuple<F, int, int>> query_k_nearest(const std::vector<Point>& points, const int k) const override {
        std::vector<std::tuple<F, int, int>> res(k, {std::numeric_limits<F>::max(), -1, -1});
        if(_num_thread_search > 1) {
            std::vector<std::tuple<F, int, int>> results(k*int(points.size()));
            ParallelFor(0, int(points.size()), _num_thread_search, [&](size_t row, size_t threadId) {
                std::priority_queue<std::pair<F, hnswlib::labeltype>> result = _alg_hnsw[_now_iter]->searchKnn(points[row].data(), k);
                for(int i = 0; i < k; i++) {
                    auto [now_eps, now_nind] = result.top();
                    result.pop();
                    results[row*k+i] = {now_eps, row, now_nind};
                }
            });
            std::sort(results.begin(), results.end());
            res.assign(results.begin(), results.begin()+k);
        }
        else {
            for(int i = 0; i < int(points.size()); i++) {
                std::priority_queue<std::pair<F, hnswlib::labeltype>> result = _alg_hnsw[_now_iter]->searchKnn(points[i].data(), k);
                std::vector<std::tuple<F, int, int>> now_res(res);
                for(int j = k; j--;) {
                    now_res.emplace_back(result.top().first, i, int(result.top().second));
                    result.pop();
                }
                std::sort(now_res.begin(), now_res.end());
                res.assign(now_res.begin(), now_res.begin()+k);
            }
        }
        for(int i = 0; i < k; i++) {
            auto &[eps2, ind, nind] = res[i];
            eps2 = sqrt(eps2);
        }

        return res;
    }
};

} // namespace QSY