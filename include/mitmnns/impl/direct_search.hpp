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
#include <limits>
#include <cmath>
#include <tuple>
#include <functional>
#include <utility>

#include <mitmnns/nearest_neighbor_search.hpp>
#include <mitmnns/gate_operation.hpp>

namespace QSY {
template <class F>
class DirectSearch : NearestNeighborSearchBase<F> {
    std::vector<int> _boundary;
    using Point = std::vector<F>;
    std::vector<Point> _points;
    int _current_boundary;
    std::function<F(const Point&, const Point&)> _dist;
  public:
    DirectSearch(std::vector<Point>& points, std::vector<int>& boundary, std::function<F(const Point&, const Point&)> dist) : _points(points), _boundary(boundary), _dist(std::move(dist)) {
        assert(_boundary[0] == 0);
        assert(_boundary[int(_boundary.size())-1] == int(points.size()));
    }

    void construct(int iter_num) override {
        _current_boundary = iter_num;
    }

    std::tuple<F, int, int> query_nearest(const std::vector<Point>& points) const override {
        F eps = std::numeric_limits<F>::max();
        int id1 = -1, id2 = -1;
        for(int i = _boundary[_current_boundary]; i < _boundary[_current_boundary+1]; i++) {
            for(int j = 0; j < int(points.size()); j++) {
                F now_eps = _dist(points[j], _points[i]);
                if(now_eps < eps) {
                    eps = now_eps;
                    id1 = j;
                    id2 = i;
                }
            }
        }
        return {eps, id1, id2};
    }

    std::vector<std::tuple<F, int, int>> query_k_nearest(const std::vector<Point>& points, const int k) const override {
        std::vector<std::tuple<F, int, int>> res(k, {std::numeric_limits<F>::max(), -1, -1});
        for(int i = _boundary[_current_boundary]; i < _boundary[_current_boundary+1]; i++) {
            for(int j = 0; j < int(points.size()); j++) {
                F now_eps = _dist(points[j], _points[i]);
                if(now_eps < std::get<0>(res[k-1])) {
                    for(int l = 0; l < k; l++) {
                        if(now_eps < std::get<0>(res[l])) {
                            res.insert(res.begin()+l, {now_eps, j, i});
                            res.pop_back();
                            break;
                        }
                    }
                }
            }
        }
        return res;
    }
};

} // namespace QSY