# Spack environment
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

Preparing the Sources
=========================

Additional to the software mentioned in README you'll need the
following programs installed on your system:

  cmake >= 3.13

Getting started
---------------

If these preliminaries are met, you should run

  dunecontrol all

which will find all installed dune modules as well as all dune modules
(not installed) which sources reside in a subdirectory of the current
directory. Note that if dune is not installed properly you will either
have to add the directory where the dunecontrol script resides (probably
./dune-common/bin) to your path or specify the relative path of the script.

Most probably you'll have to provide additional information to dunecontrol
(e. g. compilers, configure options) and/or make options.

The most convenient way is to use options files in this case. The files
define four variables:

CMAKE_FLAGS      flags passed to cmake (during configure)

An example options file might look like this:

#use this options to configure and make if no other options are given
CMAKE_FLAGS=" \
-DCMAKE_CXX_COMPILER=g++-5 \
-DCMAKE_CXX_FLAGS='-Wall -pedantic' \
-DCMAKE_INSTALL_PREFIX=/install/path" #Force g++-5 and set compiler flags

If you save this information into example.opts you can pass the opts file to
dunecontrol via the --opts option, e. g.

  dunecontrol --opts=example.opts all

More info
---------

See

     dunecontrol --help

for further options.


The full build system is described in the dune-common/doc/buildsystem (Git version) or under share/doc/dune-common/buildsystem if you installed DUNE!
