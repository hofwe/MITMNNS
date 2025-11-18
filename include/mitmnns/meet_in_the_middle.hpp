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
#include <vector>
#include <limits>
#include <cassert>
#include <tuple>

#include <mitmnns/nearest_neighbor_search.hpp>
#include <mitmnns/gate_operation.hpp>


namespace QSY {

template <class Float, class NNS, class GO>
    requires (std::floating_point<Float> &&
                DerivedFromNearestNeighborSearchBase<NNS, Float> &&
                DerivedFromGateOperationBase<GO, Float>)
class MITM {
    using Index = int;
    using Point = std::vector<Float>;
    std::vector<Point> _data1;
    std::vector<Index> _data1_boundary;
    int _data1_num, _max_count;
    NNS* _nns;
    GO _go;
    const Float FMAX = std::numeric_limits<Float>::max();

  public:
    MITM(const std::vector<Point>& data1, const std::vector<Index>& data1_boundary, NNS* nns) : _data1(data1), _data1_num(int(data1.size())), _data1_boundary(data1_boundary), _max_count(int(data1_boundary.size())-2), _nns(nns), _go() {
        assert(_data1_boundary[0] == 0);
        assert(_data1_boundary[_max_count+1] == _data1_num);
    }
    MITM(const MITM&) = delete;
    MITM& operator=(const MITM&) = delete;
    MITM(MITM&&) = default;
    MITM& operator=(MITM&&) = default;


    // input
    // point ... target gate
    // eps ... allowed error
    // output
    // tuple<Float, Index, Index>
    // Float ... error
    // Index ... index for data1
    // Index ... index for data2 (data in NNS)
    std::tuple<Float, Index, Index> eps_search(const Point& point, Float eps) const {
        std::vector<Point> targets = targetize(point, 0);
        for(int count = 1; count <= _max_count; count++) {
            _nns->construct(count);
            auto [dist1, id1, nid1] = _nns->query_nearest(targets);
            if(dist1 < eps)
                return {dist1, id1+_data1_boundary[count-1], nid1};
            targets = targetize(point, count);
            auto [dist2, id2, nid2] = _nns->query_nearest(targets);
            if(dist2 < eps)
                return {dist2, id2+_data1_boundary[count], nid2};
        }
        return {Float(-1), -1, -1};
    }

    std::tuple<Float, Index, Index> count_search(const Point& point, int count_num) const {
        assert(count_num <= _max_count*2);
        Float r_dist = FMAX;
        Index r_id = -1;
        Index r_nid = -1;
        std::vector<Point> targets = targetize(point, 0);
        for(int count = 1; count <= (count_num+1)/2; count++) {
            _nns->construct(count);
            auto [dist1, id1, nid1] = _nns->query_nearest(targets);
            if(dist1 < r_dist) {
                r_dist = dist1;
                r_id = id1+_data1_boundary[count-1];
                r_nid = nid1;
            }
            if(count*2-1 == count_num) break;
            targets = targetize(point, count);
            auto [dist2, id2, nid2] = _nns->query_nearest(targets);
            if(dist2 < r_dist) {
                r_dist = dist2;
                r_id = id2+_data1_boundary[count];
                r_nid = nid2;
            }
        }
        return {r_dist, r_id, r_nid};
    }

    std::vector<std::tuple<Float, Index, Index>> count_k_search(const Point& point, int count_num, int k) const {
        assert(count_num <= _max_count*2);
        std::vector<std::tuple<Float, Index, Index>> res(k, {FMAX, -1, -1});
        std::vector<Point> targets = targetize(point, 0);
        for(int count = 1; count <= (count_num+1)/2; count++) {
            _nns->construct(count);
            std::vector<std::tuple<Float, Index, Index>> dist_id1 = _nns->query_k_nearest(targets, k);
            for(auto& [dist, id, nid]: dist_id1) {
                id += _data1_boundary[count-1];
            }
            res = get_shortest_k(res, dist_id1, k);
            if(count*2-1 == count_num) break;
            targets = targetize(point, count);
            std::vector<std::tuple<Float, Index, Index>> dist_id2 = _nns->query_k_nearest(targets, k);
            for(auto& [dist, id, nid]: dist_id2) {
                id += _data1_boundary[count];
            }
            res = get_shortest_k(res, dist_id2, k);
        }
        return res;
    }

  private:
    std::vector<Point> targetize(const Point& point, int count) const {
        assert(count <= _max_count);
        int start = _data1_boundary[count], end = _data1_boundary[count+1];
        int res_num = end-start;
        std::vector<Point> res(res_num);
        for(int i = 0; i < res_num; i++) {
            res[i] = _go.multiply(_go.dagger(_data1[start+i]), point);
        }
        return res;
    }

    std::vector<std::tuple<Float, Index, Index>> get_shortest_k(const std::vector<std::tuple<Float, Index, Index>>& a, std::vector<std::tuple<Float, Index, Index>>& b, int k) const {
        std::vector<std::tuple<Float, Index, Index>> res(k);
        for(int i = 0, j = 0; i+j < k;) {
            if(std::get<0>(a[i]) < std::get<0>(b[j])) {
                res[i+j] = a[i];
                i++;
            }
            else {
                res[i+j] = b[j];
                j++;
            }
        }
        return res;
    }
};

} // namespace QSY