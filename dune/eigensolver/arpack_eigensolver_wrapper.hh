/*
 * Modified version of Geneo Arpack++ wrapper for supporting standard eigenvalue problems.
 */

// Header copy paste from arpack_geneo_wrapper

#ifndef DUNE_EIGENSOLVER_ARPACK_HH
#define DUNE_EIGENSOLVER_ARPACK_HH

#if HAVE_ARPACKPP

#include <cmath> // provides std::abs, std::pow, std::sqrt

#include <iostream> // provides std::cout, std::endl
#include <string>   // provides std::string

#include <algorithm>
#include <numeric>
#include <vector>

#include <dune/common/fvector.hh>    // provides Dune::FieldVector
#include <dune/common/exceptions.hh> // provides DUNE_THROW(...)

//#if DUNE_VERSION_GTE(DUNE_ISTL, 2, 8)
#include <dune/istl/blocklevel.hh>
//#endif
#include <dune/istl/bvector.hh>       // provides Dune::BlockVector
#include <dune/istl/istlexception.hh> // provides Dune::ISTLError
#include <dune/istl/io.hh>            // provides Dune::printvector(...)

//#include <dune/pdelab/backend/interface.hh>

#ifdef Status
#undef Status // prevent preprocessor from damaging the ARPACK++
              // code when "X11/Xlib.h" is included (the latter
              // defines Status as "#define Status int" and
              // ARPACK++ provides a class with a method called
              // Status)
#endif
#include "argsym.h"  // provides ARSymGenEig
#include "argnsym.h" // provides ARSymGenEig
#include "arsnsym.h" // this provides the one we need
#include "arssym.h"  // this provides the one we need 

namespace ArpackEigenSolver
{

}
