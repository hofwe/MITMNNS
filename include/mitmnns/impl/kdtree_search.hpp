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
#include <cassert>
#include <tuple>
#include <optional>
#include <cmath>
#include <limits>

#include <nanoflann.hpp>
#include <mitmnns/nearest_neighbor_search.hpp>

#include "nanoflann_utils.h"


// kdtree for single qubit gate synthesis
// using nanoflann as a kdtree package
// https://github.com/jlblancoc/nanoflann


namespace QSY {
template <class F>
class kdtree : NearestNeighborSearchBase<F> {
    PointCloud_Quat<F> _cloud;
    std::vector<int> _boundary;
    using Point = std::vector<F>;
    using SU2_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::SO3_Adaptor<F, PointCloud_Quat<F>>,
            PointCloud_Quat<F>, 4 /* dim */
            >;
    std::vector<std::optional<SU2_kd_tree_t>> _index;
    std::vector<PointCloud_Quat<F>> _now_cloud;
    int _now_iter, _current_start_ind;
  public:
    kdtree(std::vector<Point>& points, std::vector<int> boundary) : _boundary(boundary), _index(int(boundary.size())-1), _now_cloud(int(boundary.size())-1) {
        assert(_boundary[0] == 0);
        assert(_boundary[int(_boundary.size())-1] == int(points.size()));
        for(auto& p: points) {
            _cloud.pts.emplace_back(to_quat(p));
        }
    }

    void construct(int iter_num) override {
        _current_start_ind = _boundary[iter_num];
        _now_iter = iter_num;
        if(!_index[iter_num]) {
            _now_cloud[iter_num].pts.clear();
            for(int i = _current_start_ind; i < _boundary[iter_num+1]; i++) {
                _now_cloud[iter_num].pts.emplace_back(_cloud.pts[i]);
            }
            
            _index[iter_num].emplace(4 /*dim*/, _now_cloud[iter_num], 10 /* max leaf */);
        }
    }

    std::tuple<F, int, int> query_nearest(const std::vector<Point>& points) const override {
        size_t                         ret_index;
        F                              out_dist_sqr;
        nanoflann::KNNResultSet<F>     resultSet(1);

        F eps2 = std::numeric_limits<F>::max();
        int ind = -1;
        int nind = -1;
        for(int i = 0; i < int(points.size()); i++) {
            auto query = to_quat(points[i]);
            F query_pt[] = {query.w, query.x, query.y, query.z};
            resultSet.init(&ret_index, &out_dist_sqr);
            _index[_now_iter]->findNeighbors(resultSet, &query_pt[0]);
            if(out_dist_sqr < eps2) {
                eps2 = out_dist_sqr;
                ind = i;
                nind = ret_index+_current_start_ind;
            }
        }
        return {sqrt(eps2), ind, nind};
    }

    std::vector<std::tuple<F, int, int>> query_k_nearest(const std::vector<Point>& points, const int k) const override {
        uint32_t* ret_index = new uint32_t[k];
        F* out_dist_sqr = new F[k];

        std::vector<std::tuple<F, int, int>> res(k, {std::numeric_limits<F>::max(), -1, -1});
        for(int i = 0; i < int(points.size()); i++) {
            auto query = to_quat(points[i]);
            F query_pt[] = {query.w, query.x, query.y, query.z};
            int N = _index[_now_iter]->knnSearch(&query_pt[0], k, ret_index, out_dist_sqr);
            assert(N == k);
            res = get_shortest_k(res, out_dist_sqr, i, ret_index, k);
        }
        for(int i = 0; i < k; i++) {
            auto &[eps2, ind, nind] = res[i];
            eps2 = sqrt(eps2);
        }

        delete[] ret_index;
        delete[] out_dist_sqr;

        return res;
    }

  private:
    typename PointCloud_Quat<F>::Point to_quat(const std::vector<F>& su2) const {
        assert(int(su2.size()) == 8);
        typename PointCloud_Quat<F>::Point quat;
        quat.w = (su2[0] + su2[6]) * 0.5;
        quat.x = (su2[3] + su2[5]) * 0.5;
        quat.y = (su2[2] - su2[4]) * 0.5;
        quat.z = (su2[1] - su2[7]) * 0.5;
        return quat;
    }

    std::vector<std::tuple<F, int, int>> get_shortest_k(const std::vector<std::tuple<F, int, int>>& a, F* bf, int ind, uint32_t* bnind, int k) const {
        std::vector<std::tuple<F, int, int>> res(k);
        for(int i = 0, j = 0; i+j < k;) {
            if(std::get<0>(a[i]) < bf[j]) {
                res[i+j] = a[i];
                i++;
            }
            else {
                res[i+j] = {bf[j], ind, int(bnind[j])+_current_start_ind};
                j++;
            }
        }
        return res;
    }
};

} // namespace QSY