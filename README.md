![Static Badge](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)

# Multigrid DG solver
This code was developed as part of my master's assignment titled "[Towards an Efficient Multigrid Algorithm for Solving Pressure-Robust Discontinuous Galerkin Formulations of the Stokes Problem](https://essay.utwente.nl/97483/)". The goal of the research was to develop a multigrid algorithm that can be applied to the discontinuous Galerkin (DG) discretization of the Stokes problem. However, the Poisson equation was used as the model problem for the multigrid algorithm. The DG discretization was implemented for both the Poisson and the Stokes problem.

> This implementation is by no means a guideline of how a CFD program must be written, in the first place because it was written in Python. Moreover, the code was developed in an engineering context and not in the field of computer science. Of course, best practices in coding were followed as much as possible.

## Installation
The code can be installed by cloning this repository and running `python -m pip install -r requirements.txt` to install the dependencies. The code was written in Python 3.10 and should therefore be compatible with Python versions `>=3.10`.

## Usage
The case needs to be set up in a paramfile. In this file, information about the grid, solver, problem, visualization and some other things need to be set. Also, ParaView can be opened automatically to visualize the results by specifying the location to the ParaView executable in the paramfile. An example for the paramfile is given in `paramfile.yml`.

Once the case is set up using the paramfile, the program can be run by executing `python -m dgfem` and specifying an option for the solver (see below). For example, if the multigrid solver is set up in the paramfile (only possible for the Poisson problem), solving the case using the multigrid algorithm can be done by `python -m dgfem -m`.


```
usage: DG solver [-h] [--grid-folder GRID_FOLDER] [-f GRID_FILE] [--p-grid P_GRID] [--p-solution P_SOLUTION] [-d] [-s] [--smoother SMOOTHER] [-amg] [-k] [-m]
                 [-fvm] [-amp] [--dg-discretization] [--fvm-discretization] [--check-eigenvalues] [--check-condition-number] [--plot-sparsity-pattern] [-v]
                 [--silent]

DG solver for the Poisson and Stokes problems

options:
  -h, --help            show this help message and exit
  --grid-folder GRID_FOLDER
  -f GRID_FILE, --grid-file GRID_FILE
  --p-grid P_GRID
  --p-solution P_SOLUTION
  -d, --solve-direct
  -s, --solve-smoother  mutually inclusive with --smoother
  --smoother SMOOTHER
  -amg, --solve-pyamg
  -k, --solve-krylov
  -m, --solve-multigrid
  -fvm, --solve-finite-volume-method
  -amp, --solve-smoother-amplification
                        mutually inclusive with --fvm-discretization or --dg-discretization
  --dg-discretization
  --fvm-discretization
  --check-eigenvalues
  --check-condition-number
  --plot-sparsity-pattern
  -v, --verbose
  --silent
```

## Known issues/limitations
- The multigrid algorithm is only implemented for the Poisson problem, not for the Stokes problem.
- Caching is based on pickling. Therefore it is not suitable to cache large problems. It would be better to implement caching in binary format, for example using [hdf5py](https://docs.h5py.org/en/stable/).
- The results of the Stokes problem in curvilinear grids contain pressure spikes. The origin of this is unknown at the moment. More information about this can be found in the thesis.

## License
This project is licensed under the GNU General Public License v3.0, so feel free to contribute!