# Mixed-precision iterative refinement for the low-rank Lyapunov equation

This repository contains the code used in the technical report:

P. Benner and X. Liu. *Mixed-precision iterative refinement for the low-rank Lyapunov equation*, October 2025.

## Dependencies

The code was developed in MATLAB R2024b. The following MATLAB toolbox is used in the code:

- [Chop](https://github.com/higham/chop)

## Directory structure

The repository contains the following subfolders:

- `external/`, which contains external scripts/functions required by the code;
- `include/`, which contains the main subroutines required to reproduce the results; 
- `testmats/`, which contains the tested problems from the [SLICOT](https://www.slicot.org) library.

The main executable scripts are located in the root directory:

- `simple_test.m`, a simple test;
- `test_slicot.m`, which generates the data presented in Table 4.3; and
- `test_synthetic.m`, which generates the data presented in Table 4.1.

The following algorithms in the manuscript are implemented:

- Algorithm 2.3 and Algorithm 2.6 with sign functino Newton iteration as the solver are implemented in `include/lyap_snir.m`;
- Algorithm 3.1 and Algorithm 3.2 are implemented in `include/lyap_sn.m`.

## License

This project is licensed under the terms of the BSD 2-Clause "Simplified" License (SPDX-License-Identifier: BSD-2-Clause).
