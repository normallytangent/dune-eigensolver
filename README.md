DUNE-library
============

This package contains some iterative eigenvalue solvers to be used by the  numerical methods in DUNE.

Dependencies
============
dune-eigensolver depends on the following software packages
- C++ compiler
- CMake >= 3.13
- oneMKL
- oneTBB
- UMFPACK
- [VCL](https://github.com/vectorclass/version2) 2.0
- [Eigen](https://eigen.tuxfamily.org/) 3.3
- [ARPACK-NG](https://github.com/opencollab/arpack-ng)
- [ARPACK++](https://github.com/m-reuter/arpackpp)
- OpenMPI

Additionally, it depends on the following DUNE packages
- dune-common 2.9
- dune-istl 2.9

A quick way to resolve dependencies is to set up a Spack environment - the following setup was tested on Milan and is the recommended approach.

Spack environment
=================
Setting up a local spack enviroment for dune-eigensolver on Milan. The packages' versions correspond to the dependency of the repository as on the branch `develop`.
1. Create a local directory, spack-env and copy the commands below to `spack-env/spack.yaml` file. The comands describe a set of packages to be installed, along with configuration settings.
```
spack:
  # add package specs to the `specs` list
  specs:
  - cmake@3.27.9%gcc@12.2.0
  - intel-oneapi-tbb@2021.12.0%gcc@12.2.0
  - openmpi@5.0.3%gcc@12.2.0
  - suite-sparse@5.13.0%gcc@12.2.0
  - arpack-ng@3.9.0%gcc@12.2.0
  view:
    default:
      root: .spack-env/view
      link: roots
  concretizer:
    unify: false
```
2. Install spack environment
```
$ spack env activate --create /path/to/spack-env
```
3. To load the environment on next login
```
$ spacktivate /path/to/spack-env
```

Installation
============
The installation of dune-eigensolver maintains an out-of-source build in order to keep the source controlled tree intact while allowing the user the flexibility of removing and reinstalling the complete build. The source tree also maintains a local directory to some of the package's dependencies.

1. Make a separate directory for a local installation of DUNE packages and dependencies.
```
$ mkdir -p DUNE/external && cd DUNE/external
```
2. Resolve all the dependencies, and clone and install VCL, Eigen inside `external/`.
3. Return to `DUNE/`and set some CMake variables and compiler flags in a file `release.opts`, or optionally through the variable `$OPTSFILE`. Modify the paths as needed. 
```
CMAKE_FLAGS="
    -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -g0 -funroll-loops -ftemplate-depth=5120 -march=native -Wa,-q'
    -DCMAKE_BUILD_TYPE=Release
    -DDUNE_SYMLINK_TO_SOURCE_TREE=1
    -DDUNE_ENABLE_PYTHONBINDINGS=NO
    -DARPACKPP_INCLUDE_DIR=/path/to/arpackpp/include/arpackpp/
    -DVCL_ROOT=/path/to/vcl
    -DEIGEN_ROOT=/path/to/eigen/include/eigen3/Eigen
  "
```

4. Clone dune-common, dune-istl

5. Clone dune-eigensolver
```
$ git clone https://github.com/normallytangent/dune-eigensolver.git
```

Optionally,  If you want to test dune-eigensolver with the Aharmonic subdomains, clone also the submodule Sampler under `dune-eigensolver/src/sampler`. Please note that because of the matrices, Sampler can get quite large.
```
$ submodule update --init --recursive
```
6. Setup script for DUNE installation
```
#!/bin/bash
ROOT=$(pwd)
BUILDDIR=$ROOT/release-build
OPTSFILE=release.opts
MODULES="common istl eigensolver"
for i in $MODULES; do
    echo build $i
    ./dune-common/bin/dunecontrol --builddir=$BUILDDIR  --opts=$OPTSFILE --only=dune-$i all
done
```
7. ... and run. This will create a separate build directory, `release-build` under `DUNE/`.

More info
---------

See
```
$ .dune-common/bin/dunecontrol --help
```
for further options.


The full build system is described in the dune-common/doc/buildsystem (Git version) or under share/doc/dune-common/buildsystem if you installed DUNE!


<!-- Example run -->
<!-- =========== -->

<!-- Links
-----
[VCL]: https://github.com/vectorclass/version2 
[Eigen]: https://eigen.tuxfamily.org/
[ARPACK-NG]: https://github.com/opencollab/arpack-ng
[ARPACK++]: https://github.com/m-reuter/arpackpp -->
