# MITMNNS: Meet-in-the-Middle + Nearest Neighbor Search for Quantum Compilation

This project provides an efficient algorithmic framework for **quantum gate synthesis** that combines **Meet-in-the-Middle (MITM)** search with **Nearest Neighbor Search (NNS)** or **Approximate Nearest Neighbor Search (ANNS)** techniques.
It can be applied to arbitrary gate sets such as **Clifford+T** and **Clifford+V**, enabling users to perform optimal unitary decompositions *quadratically faster* than exhaustive search.

For theoretical background, see the reference:
[arXiv:2510.08312](https://arxiv.org/abs/2510.08312)

---

## Installation

MITMNNS is a **header-only C++ library**, requiring no additional linking.

```bash
$ git clone <repository-url>
$ cd MITMNNS
$ cmake -B build -S .
$ sudo cmake --install build
```

---

## Usage

### Overview

This algorithm accelerates exhaustive search using the **Meet-in-the-Middle (MITM)** approach.
Within a given gate set:

* **Base gates** are the gates concatenated during synthesis.
* **Suffix gates** are appended at the end.

(e.g., for n-qubit Clifford+T, the base gates are $\pi/4$ rotations around each non-identity Pauli operator $P\in P_n$, denoted as $R_P(\pi/4)$, while n-qubit Clifford gates act as suffixes.)

When searching up to a total count of **2m** base gates, the search space is split in half:

* **data1** — exhaustive search region
* **data2** — explored using NNS/ANNS

Both contain elements up to count `m`, and `data2` additionally includes suffix gates.

These datasets (`data1`, `data2`) must be prepared as `std::vector`s ordered by increasing gate count, with separate `std::vector`s specifying their count boundaries.


### 3-Step Setup

1. **Nearest Neighbor Search Class**

   Implement a class that inherits from
   [`NearestNeighborSearchBase`](include/mitmnns/nearest_neighbor_search.hpp),
   providing your preferred search method (e.g., KD-tree, HNSW).

   * See examples in `include/mitmnns/impl/`:

     * `direct_search.hpp` (exhaustive)
     * `kdtree_search.hpp` (KD-tree using [nanoflann](https://github.com/jlblancoc/nanoflann))
     * `hnsw_search.hpp` (HNSW using [HNSWlib](https://github.com/nmslib/hnswlib); supports multithreading by default with 20 threads)

2. **Gate Operation Class**

   Create a class that inherits from
   [`gate_operation.hpp`](include/mitmnns/gate_operation.hpp),
   defining the required gate operations.

   * Examples:

     * `SU2_operation.hpp` — SU(2) gates
     * `SUD_operation.hpp` — SU(D) gates
     * `conditionally_controlled_gate_operation.hpp` — subgroup-guided search (as in [arXiv:2510.08312](https://arxiv.org/abs/2510.08312))

3. **Perform MITM Search**

   Instantiate the [`MITM`](include/mitmnns/meet_in_the_middle.hpp) class
   with your NNS/ANNS implementation, gate operations, and the prepared datasets.
   You can then execute following searches:

   * `eps_search` — find the smallest-count synthesis within given error ε
   * `count_search` — minimize error under a given count limit
   * `count_k_search` — find the top k lowest-error synthesis within count limit m

---

## Theoretical Foundation

* **MITM Search:** Splits the exponential synthesis space to achieve *quadratic* speedup compared to brute-force enumeration.
* **NNS/ANNS Integration:** Uses geometric proximity in SU(n) to guide synthesis efficiently.
* **Subgroup-Guided Search:** Allows selective exploration of structured unitary subgroups (e.g., controlled-gate hierarchies).

For detailed formulation and performance analysis, see
[arXiv:2510.08312](https://arxiv.org/abs/2510.08312).

---

## Examples

Example implementations are provided in the `examples/` directory.
They demonstrate approximate synthesis of Haar-random unitaries.
It is required to install [nanoflann](https://github.com/jlblancoc/nanoflann) and/or [HNSWlib](https://github.com/nmslib/hnswlib) beforehand if you want to run files with kdtree and/or HNSW respectively.

After installation:

```bash
$ cd examples
$ cmake -B build -S .
$ cmake --build build
```

Run:

For Mac OS or Linux
```bash
$ ./build/<executable file>
```

### Example Programs

| File                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `direct_V.cpp`        | 1-qubit Clifford+V synthesis using exhaustive search          |
| `kdtree_V.cpp`        | 1-qubit Clifford+V synthesis using KD-tree                    |
| `hnsw_V.cpp`          | 1-qubit Clifford+V synthesis using HNSW                       |
| `hnsw_T.cpp`          | 2-qubit Clifford+T synthesis using single-threaded HNSW       |
| `hnsw_mt_T.cpp`       | 2-qubit Clifford+T synthesis using multi-threaded HNSW        |
| `hnsw_Clifford_T.cpp` | 2-qubit Clifford+T synthesis using HNSW with Clifford ignoring distance |
| `hnsw_CC_V.cpp`       | 2-qubit Clifford+V synthesis using subgroup-guided search     |

---

### Notes

* Example searches use **phase-invariant norm** (distance modulo a global phase factor).
* HNSW search graphs are saved to `examples/data/` after first run, enabling faster subsequent execution.
* Haar-random unitaries are stored in `examples/data/` as
  `SU{D}_{100 or C100}.fvecs`, where:

  * `D` = unitary dimension (2ⁿ)
  * `100` = number of random unitaries
  * `C100` = number of random *controlled* unitaries

* Basically, Pauli matrices are used as suffixes, since including the full Clifford group would easily exceed memory capacity (cf. 11520 2-qubit Clifford gates). By using ANNS in distance calculation, we can use a distance function which identifies Clifford gates. The function is used in `hnsw_Clifford_T.cpp`.

Currently, data generators are available up to 3 qubits.
For higher dimensions, users can implement their own generators.

---

## Contributing

Contributions are welcome!
If you’d like to improve algorithms, add nearest neighbor searches, or enhance documentation:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit and push changes.
4. Open a Pull Request.

Please follow the existing code style and include tests where appropriate.

---

## Contact

Maintained by **Soichiro Yamazaki**
Graduate School of Science, The University of Tokyo
Email: soichiro.yamazaki@phys.s.u-tokyo.ac.jp
GitHub: [hofwe](https://github.com/hofwe)

