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
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <string>
#include <utility>
#include <cmath>
#include <cstdint>

#include <mitmnns/impl/SU2_operation.hpp>
#include <mitmnns/impl/SUD_operation.hpp>
#include <mitmnns/impl/conditionally_controlled_gate_operation.hpp>
#include <mitmnns/impl/hnswlib_utils.h>

void load_fvecs(const char* filename, float*& data, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

    in.seekg(0, std::ios::beg);
    for(size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i*dim), dim*4);
    }
    in.close();
}

void save_fvecs(const std::string& filename, const std::vector<std::vector<float>>& data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("cannot open file: " + filename);

    for (const auto& vec : data) {
        int32_t dim = static_cast<int32_t>(vec.size());
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
        ofs.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
    }
}

int get_dim(int whole_dim) {
    int u = 0;
    while(u*u*2 < whole_dim) u++;
    assert(u*u*2 == whole_dim);
    return u;
}
inline int to_real(int i, int j, int dim) {
    return (i*dim+j)*2;
}
inline int to_imag(int i, int j, int dim) {
    return (i*dim+j)*2+1;
}
template <class F>
std::vector<F> tensor(const std::vector<F>& a, const std::vector<F>& b) {
    int n = a.size(), m = b.size();
    assert(!((n|m)&1));
    int nm = n*m/2;
    std::vector<F> res(nm);
    int ad = get_dim(n), bd = get_dim(m), cd = ad*bd;
    for(int i = 0; i < ad; i++) {
        for(int j = 0; j < ad; j++) {
            for(int k = 0; k < bd; k++) {
                for(int l = 0; l < bd; l++) {
                    res[to_real(i*bd+k, j*bd+l, cd)] = a[to_real(i, j, ad)]*b[to_real(k, l, bd)] - a[to_imag(i, j, ad)]*b[to_imag(k, l, bd)];
                    res[to_imag(i*bd+k, j*bd+l, cd)] = a[to_real(i, j, ad)]*b[to_imag(k, l, bd)] + a[to_imag(i, j, ad)]*b[to_real(k, l, bd)];
                }
            }
        }
    }
    return res;
}

// for SU(D)
template <class F, int D>
class V_gen {
    using Gate = std::vector<F>;
    std::vector<int> _par, _prev;
    // long long only allows us up to 3-qubit
    std::vector<unsigned long long> _avail;
    int _max_count;
    QSY::SUD<F, D> _go;
  public:
    std::array<Gate, D*D*2> _Vs;
    std::vector<Gate> _data;
    std::vector<int> _bound;

    V_gen(const std::vector<Gate>& start, int max_count, size_t num_thread=20) : _par(int(start.size()), -1), _prev(int(start.size()), -1), _avail(int(start.size()), (unsigned long long)(-1LL)), _max_count(max_count), _data(start), _bound({0, int(start.size())}) {
        for(auto& st: start) {
            assert(int(st.size()) == D*D*2);
        }
        _Vs.fill(Gate(D*D*2));

        Gate I(8);
        I[0] = I[6] = 1.;
        Gate X(8);
        X[2] = X[4] = 1.;
        Gate Y(8);
        Y[3] = -1.;
        Y[5] = 1.;
        Gate Z(8);
        Z[0] = 1.;
        Z[6] = -1.;
        std::array<Gate, 4> Pauli({I, X, Y, Z});
        const static F cos_theta = sqrt(0.2f), sin_theta = 2.*cos_theta;
        for(int i = 0; i < D*D; i++) {
            Gate now = Gate({F(1.), F(0.)});
            for(int j = 0; (1<<j) < D; j++) {
                now = tensor(Pauli[(i>>(j*2))&3], now);
            }
            for(int j = 0; j < D; j++) {
                for(int k = 0; k < D; k++) {
                    _Vs[i*2][to_real(j, k, D)] = -sin_theta*now[to_imag(j, k, D)];
                    _Vs[i*2][to_imag(j, k, D)] = sin_theta*now[to_real(j, k, D)];
                    if(j == k) {
                        _Vs[i*2][to_real(j, k, D)] += cos_theta;
                    }
                }
            }
            _Vs[i*2+1] = _go.dagger(_Vs[i*2]);
        }
        std::vector<std::vector<bool>> commute(D*D, std::vector<bool>(D*D));
        for(int i = 1; i < D*D; i++) {
            for(int j = 1; j < D*D; j++) {
                int u = 1;
                for(int k = 0; (1<<k) < D; k++) {
                    int ik = i>>(k*2)&3;
                    int jk = j>>(k*2)&3;
                    u ^= int(ik != jk and ik*jk);
                }
                commute[i][j] = u;
            }
        }
        for(int c = 1; c <= max_count; c++) {
            for(int i = _bound[c-1]; i < _bound[c]; i++) {
                for(int j = 2; j < D*D*2; j++) if(_avail[i]>>(j/2)&1) {
                    if(_par[i] == (j^1)) continue;
                    _par.emplace_back(j);
                    _prev.emplace_back(i);
                    // _data.emplace_back(_go.multiply(_Vs[j], _data[i]));
                    unsigned long long av = 0ULL;
                    for(int k = 0; k < D*D; k++) {
                        if((k > j/2 and (!commute[j/2][k] or (_avail[i]>>k&1))) or (k < j/2 and !commute[k][j/2]) or k == j/2) {
                            av |= (1<<k);
                        }
                    }
                    _avail.emplace_back(av);
                }
            }
            _data.resize(int(_par.size()));
            ParallelFor(_bound[c], int(_par.size()), num_thread, [&](size_t row, size_t threadId) {
                _data[row] = _go.multiply(_Vs[_par[row]], _data[_prev[row]]);
            });
            _bound.emplace_back(int(_data.size()));
        }
    }

    std::pair<std::vector<int>, int> decode(int id) const {
        if(id < 0) {
            return {std::vector<int>(), -1};
        }

        std::vector<int> gates;
        while(_par[id] >= 0) {
            gates.emplace_back(_par[id]);
            id = _prev[id];
        }
        return {gates, id};
    }

    Gate reconstruct(const std::pair<std::vector<int>, int>& sequence) const {
        if(sequence.second < 0) {
            return Gate();
        }
        Gate g = _data[sequence.second];
        int count = sequence.first.size();
        for(int i = count; i--;) {
            g = _go.multiply(_Vs[sequence.first[i]], g);
        }
        return g;
    }

    static void output_line(const std::vector<int>& line) {
        if(line.empty()) {
            std::cout << "No line available ";
        }
        const std::array<std::string, 4> Pauli_string({"i", "x", "y", "z"});
        std::array<std::string, D*D*2> Vs_string({});
        Vs_string.fill("V");
        for(int i = 0; i < D*D; i++) {
            for(int j = 0; (1<<j) < D; j++) {
                Vs_string[i*2] += Pauli_string[(i>>(j*2))&3];
            }
            std::reverse(Vs_string[i*2].begin()+1, Vs_string[i*2].end());
            Vs_string[i*2+1] = Vs_string[i*2] + std::string("†");
        }
        for(auto i: line) {
            std::cout << Vs_string[i] << ' ';
        }
    }
};

template <int D>
void output(int i, float e, int id1, int id2, const std::vector<float>& query, const V_gen<float, D>& v1, const V_gen<float, D>& v2) {
    std::cout << i << std::endl;
    std::cout << "error: " << e << std::endl;
    auto l1 = v1.decode(id1), l2 = v2.decode(id2);
    if(id1 < 0) return;
    std::cout << "count: " << int(l1.first.size()) << '+' << int(l2.first.size()) << std::endl;
    V_gen<float, D>::output_line(l1.first);
    V_gen<float, D>::output_line(l2.first);
    std::cout << "\nquery:\n";
    for(int j = 0; j < D*D*2; j++) std::cout << query[j] << ' ';
    auto g1 = v1.reconstruct(l1), g2 = v2.reconstruct(l2);
    auto g = QSY::SUD<float, D>().multiply(g1, g2);
    std::cout << "\napprox:\n";
    for(int j = 0; j < D*D*2; j++) std::cout << g[j] << ' ';
    std::cout << "\n" << std::endl;
}

// SU(D)
template <class F, int D>
class T_gen {
    using Gate = std::vector<F>;
    std::vector<int> _par, _prev;
    // long long only allows us up to 3-qubit
    std::vector<unsigned long long> _avail;
    int _max_count;
    QSY::SUD<F, D> _go;
  public:
    std::array<Gate, D*D> _Ts;
    std::vector<Gate> _data;
    std::vector<int> _bound;

    T_gen(const std::vector<Gate>& start, int max_count, size_t num_thread=20) : _par(int(start.size()), -1), _prev(int(start.size()), -1), _avail(int(start.size()), (unsigned long long)(-1LL)), _max_count(max_count), _data(start), _bound({0, int(start.size())}), _Ts{} {
        for(auto& st: start) {
            assert(int(st.size()) == D*D*2);
        }
        _Ts.fill(Gate(D*D*2));

        Gate I(8);
        I[0] = I[6] = 1.;
        Gate X(8);
        X[2] = X[4] = 1.;
        Gate Y(8);
        Y[3] = -1.;
        Y[5] = 1.;
        Gate Z(8);
        Z[0] = 1.;
        Z[6] = -1.;
        std::array<Gate, 4> Pauli({I, X, Y, Z});
        const static F theta = M_PI*0.125;
        const static F cos_theta = cos(theta), sin_theta = sin(theta);
        for(int i = 0; i < D*D; i++) {
            Gate now = Gate({F(1.), F(0.)});
            for(int j = 0; (1<<j) < D; j++) {
                now = tensor(Pauli[(i>>(j*2))&3], now);
            }
            for(int j = 0; j < D; j++) {
                for(int k = 0; k < D; k++) {
                    _Ts[i][to_real(j, k, D)] = sin_theta*now[to_imag(j, k, D)];
                    _Ts[i][to_imag(j, k, D)] = -sin_theta*now[to_real(j, k, D)];
                    if(j == k) {
                        _Ts[i][to_real(j, k, D)] += cos_theta;
                    }
                }
            }
        }
        // for(int i = 0; i < D*D; i++) {
        //     for(int j = 0; j < D*D*2; j+=2) {
        //         F re = _Ts[i][j]*cos_theta - _Ts[i][j+1]*sin_theta;
        //         F im = _Ts[i][j]*sin_theta + _Ts[i][j+1]*cos_theta;
        //         _Ts[i][j] = re;
        //         _Ts[i][j+1] = im;
        //     }
        // }
        std::vector<std::vector<bool>> commute(D*D, std::vector<bool>(D*D));
        for(int i = 1; i < D*D; i++) {
            for(int j = 1; j < D*D; j++) {
                int u = 1;
                for(int k = 0; (1<<k) < D; k++) {
                    int ik = i>>(k*2)&3;
                    int jk = j>>(k*2)&3;
                    u ^= int(ik != jk and ik*jk);
                }
                commute[i][j] = u;
            }
        }
        for(int c = 1; c <= max_count; c++) {
            for(int i = _bound[c-1]; i < _bound[c]; i++) {
                for(int j = 1; j < D*D; j++) if(_avail[i]>>j&1) {
                    _par.emplace_back(j);
                    _prev.emplace_back(i);
                    // _data.emplace_back(_go.multiply(_Ts[j], _data[i]));
                    unsigned long long av = 0ULL;
                    for(int k = 0; k < D*D; k++) {
                        if((k > j and (!commute[j][k] or (_avail[i]>>k&1))) or (k < j and !commute[k][j])) {
                            av |= (1<<k);
                        }
                    }
                    _avail.emplace_back(av);
                }
            }
            _data.resize(int(_par.size()));
            ParallelFor(_bound[c], int(_par.size()), num_thread, [&](size_t row, size_t threadId) {
                _data[row] = _go.multiply(_Ts[_par[row]], _data[_prev[row]]);
            });
            _bound.emplace_back(int(_data.size()));
        }
    }

    std::pair<std::vector<int>, int> decode(int id) const {
        if(id < 0) {
            return {std::vector<int>(), -1};
        }

        std::vector<int> gates;
        while(_par[id] >= 0) {
            gates.emplace_back(_par[id]);
            id = _prev[id];
        }
        return {gates, id};
    }

    Gate reconstruct(const std::pair<std::vector<int>, int>& sequence) const {
        if(sequence.second < 0) {
            return Gate();
        }
        Gate g = _data[sequence.second];
        int count = sequence.first.size();
        for(int i = count; i--;) {
            g = _go.multiply(_Ts[sequence.first[i]], g);
        }
        return g;
    }

    static void output_line(const std::vector<int>& line) {
        if(line.empty()) {
            std::cout << "No line available ";
        }
        const std::array<std::string, 4> Pauli_string({"i", "x", "y", "z"});
        std::array<std::string, D*D> Ts_string({});
        Ts_string.fill("T");
        for(int i = 0; i < D*D; i++) {
            for(int j = 0; (1<<j) < D; j++) {
                Ts_string[i] += Pauli_string[(i>>(j*2))&3];
            }
            std::reverse(Ts_string[i].begin()+1, Ts_string[i].end());
        }
        for(auto i: line) {
            std::cout << Ts_string[i] << ' ';
        }
    }
};

template <int D>
void output(int i, float e, int id1, int id2, const std::vector<float>& query, const T_gen<float, D>& t1, const T_gen<float, D>& t2) {
    std::cout << i << std::endl;
    std::cout << "error: " << e << std::endl;
    auto l1 = t1.decode(id1), l2 = t2.decode(id2);
    if(id1 < 0) return;
    std::cout << "count: " << int(l1.first.size()) << '+' << int(l2.first.size()) << std::endl;
    T_gen<float, D>::output_line(l1.first);
    T_gen<float, D>::output_line(l2.first);
    std::cout << "\nquery:\n";
    for(int j = 0; j < D*D*2; j++) std::cout << query[j] << ' ';
    auto g1 = t1.reconstruct(l1), g2 = t2.reconstruct(l2);
    auto g = QSY::SUD<float, D>().multiply(g1, g2);
    std::cout << "\napprox:\n";
    for(int j = 0; j < D*D*2; j++) std::cout << g[j] << ' ';
    std::cout << "\n" << std::endl;
}



// for CC(A, B)
template <class F>
class CC_V_gen {
    using Gate = std::vector<F>;
    std::vector<int> _par, _prev;
    // long long only allows us up to 3-qubit
    int _max_count;
    QSY::SU2<F> _go;
  public:
    std::array<Gate, 12> _Vs;
    std::vector<Gate> _data;
    std::vector<int> _bound;

    CC_V_gen(const std::vector<Gate>& start, int max_count, size_t num_thread=20) : _par(int(start.size()), -1), _prev(int(start.size()), -1), _max_count(max_count), _data(start), _bound({0, int(start.size())}) {
        for(auto& st: start) {
            assert(int(st.size()) == 16);
        }
        _Vs.fill(Gate());

        Gate X(8);
        X[3] = X[5] = 1.;
        Gate Y(8);
        Y[2] = 1.;
        Y[4] = -1.;
        Gate Z(8);
        Z[1] = 1.;
        Z[7] = -1.;
        const static F cos_theta = sqrt(0.2f), sin_theta = 2.*cos_theta;
        for(int i = 0; i < 3; i++) {
            Gate now;
            switch(i) {
                case 0:
                    now = X;
                    break;
                case 1:
                    now = Y;
                    break;
                default:
                    now = Z;
            }
            for(int i = 0; i < 8; i++) {
                now[i] *= sin_theta;
            }
            now[0] = now[6] = cos_theta;
            auto _now = _go.dagger(now);
            for(int j = 0; j < 4; j++) {
                for(int k = 0; k < 2; k++) {
                    if(j>>k&1) {
                        _Vs[i*4+j].insert(_Vs[i*4+j].end(), _now.begin(), _now.end());
                    }
                    else {
                        _Vs[i*4+j].insert(_Vs[i*4+j].end(), now.begin(), now.end());
                    }
                }
            }
        }
        // Gate Vx(8);
        // Vx[0] = Vx[6] = 1.;
        // Vx[3] = Vx[5] = 2.;
        // for(int i = 0; i < 8; i++) {
        //     Vx[i] *= sqrt(0.2);
        // }
        // Gate Vy(8);
        // Vy[0] = Vy[6] = 1.;
        // Vy[2] = 2.;
        // Vy[4] = -2.;
        // for(int i = 0; i < 8; i++) {
        //     Vy[i] *= sqrt(0.2);
        // }
        // Gate Vz(8);
        // Vz[0] = Vz[6] = 1.;
        // Vz[1] = 2.;
        // Vz[7] = -2.;
        // for(int i = 0; i < 8; i++) {
        //     Vz[i] *= sqrt(0.2);
        // }
        // for(int i = 0; i < 16; i++) {
        //     std::cout << _Vs[0][i] << ' ' << Vx[i%8] << ' ' << _Vs[4][i] << ' ' << Vy[i%8] << ' ' << _Vs[8][i] << ' ' << Vz[i%8] << std::endl;
        // }
        for(int c = 1; c <= max_count; c++) {
            for(int i = _bound[c-1]; i < _bound[c]; i++) {
                for(int j = 0; j < 12; j++) {
                    if((_par[i]^j) == 3) continue;
                    if(_par[i]>j and !((_par[i]^j)>>2)) continue;
                    _par.emplace_back(j);
                    _prev.emplace_back(i);
                    // _data.emplace_back(Gate());
                    // for(int k = 0; k < 2; k++) {
                    //     Gate v(8), d(8);
                    //     v.copy(v.begin(), v.end(), _Vs[j].begin()+8*k);
                    //     d.copy(d.begin(), d.end(), _data[j].begin()+8*k);
                    //     Gate m = _go.multiply(v, d);
                    //     _data.rbegin()->insert(_data.rbegin()->end(), m.begin(), m.end());
                    // }
                }
            }
            _data.resize(int(_par.size()));
            ParallelFor(_bound[c], int(_par.size()), num_thread, [&](size_t row, size_t threadId) {
                _data[row].clear();
                for(int k = 0; k < 2; k++) {
                    Gate v(8), d(8);
                    std::copy(_Vs[_par[row]].begin()+8*k, _Vs[_par[row]].begin()+8*(k+1), v.begin());
                    std::copy(_data[_prev[row]].begin()+8*k, _data[_prev[row]].begin()+8*(k+1), d.begin());
                    Gate m = _go.multiply(v, d);
                    _data[row].insert(_data[row].end(), m.begin(), m.end());
                }
            });
            _bound.emplace_back(int(_data.size()));
        }
    }

    std::vector<Gate> generate_sa() {
        std::vector<Gate> sa(*_bound.rbegin());
        ParallelFor(0, *_bound.rbegin(), 20, [&](size_t row, size_t threadId) {
            Gate a(_data[row].begin(), _data[row].begin()+8);
            Gate b(_data[row].begin()+8, _data[row].end());
            sa[row] = _go.multiply(_go.dagger(a), b);
        });
        return sa;
    }

    std::pair<std::vector<int>, int> decode(int id) const {
        if(id < 0) {
            return {std::vector<int>(), -1};
        }

        std::vector<int> gates;
        while(_par[id] >= 0) {
            gates.emplace_back(_par[id]);
            id = _prev[id];
        }
        return {gates, id};
    }

    Gate reconstruct(const std::pair<std::vector<int>, int>& sequence, bool sa = false) const {
        if(sequence.second < 0) {
            return Gate();
        }
        Gate g = _data[sequence.second];
        int count = sequence.first.size();
        for(int i = count; i--;) {
            Gate now;
            for(int k = 0; k < 2; k++) {
                Gate v(8), d(8);
                std::copy(_Vs[sequence.first[i]].begin()+8*k, _Vs[sequence.first[i]].begin()+8*(k+1), v.begin());
                std::copy(g.begin()+8*k, g.begin()+8*(k+1), d.begin());
                Gate m = _go.multiply(v, d);
                now.insert(now.end(), m.begin(), m.end());
            }
            g = std::move(now);
        }
        if(sa) {
            Gate a(g.begin(), g.begin()+8);
            Gate b(g.begin()+8, g.end());
            return _go.multiply(_go.dagger(a), b);
        }
        return g;
    }

    static void output_line(const std::vector<int>& line) {
        if(line.empty()) {
            std::cout << "No line available ";
        }
        std::array<std::string, 12> Vs_string({"Vix", "Vzx†", "Vzx", "Vix†", "Viy", "Vzy†", "Vzy", "Viy†", "Viz", "Vzz†", "Vzz", "Viz†"});
        for(auto i: line) {
            std::cout << Vs_string[i] << ' ';
        }
    }
};

void output(int i, float e, int id1, int id2, const std::vector<float>& query, const CC_V_gen<float>& v1, const CC_V_gen<float>& v2) {
    std::cout << i << std::endl;
    std::cout << "error: " << e << std::endl;
    auto l1 = v1.decode(id1), l2 = v2.decode(id2);
    if(id1 < 0) return;
    std::cout << "count: " << int(l2.first.size()) << '+' << int(l1.first.size()) << std::endl;
    CC_V_gen<float>::output_line(l2.first);
    CC_V_gen<float>::output_line(l1.first);
    std::cout << "\nquery:\n";
    for(int j = 0; j < 8; j++) std::cout << query[j] << ' ';
    auto g1 = v1.reconstruct(l1), g2 = v2.reconstruct(l2, true);
    auto g11 = QSY::SU2<float>().dagger(g1);
    std::copy(g11.begin(), g11.end(), g1.begin());
    auto g = QSY::SU2ConditionallyControlledGate<float>().multiply(g1, g2);
    std::cout << "\napprox:\n";
    for(int j = 0; j < 8; j++) std::cout << g[j] << ' ';
    std::cout << "\n" << std::endl;
}