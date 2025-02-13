// C++ includes
#include <math.h>
#include <iostream>
#include <vector>
#include <iomanip>

#define SIZE 8

#ifndef NUM
#define NUM SIZE
#endif

std::vector<double> eigenvalues_laplace_dirichlet_2d(int N)
{
  std::vector<double> ev(N * N);
  double h = 1 / (N + 1.0);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      ev[j * N + i] = 4.0 * (std::sin(0.5 * h * (i + 1) * M_PI) * std::sin(0.5 * h * (i + 1) * M_PI) + std::sin(0.5 * h * (j + 1) * M_PI) * std::sin(0.5 * h * (j + 1) * M_PI));

  std::sort(ev.begin(), ev.end(), std::greater{});
  return ev;
}

std::vector<double> eigenvalues_laplace_neumann_2d(int N)
{
  std::vector<double> ev(N * N);
  double h = 1 / (N + 1.0);
  
  // https://youtu.be/ik2_5QVVLLA?si=IDc37pJrnrah7PKC
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
//      ev[j * N + i] = 8.0 * (std::cosh(h * (2j - 1)) * std::cosh(h * (2i - 1)) / std::sinh(h * (j + 1) * M_PI) * std::cosh(h * (j + 1) * M_PI));
  
 // https://math.stackexchange.com/a/802003/288799 
 // for (std::size_t i = 0; i < N; ++i)
 //   for (std::size_t j = 0; j < N; ++j)
 //     ev[j * N + i] = 4.0 * (std::cosh(h * (i + 1) * M_PI) * std::cosh(h * (i + 1) * M_PI) + std::cosh(h * (j + 1) * M_PI) * std::cosh(h * (j + 1) * M_PI));

  std::sort(ev.begin(), ev.end(), std::greater{});
  return ev;
}

int main() {
  
  std::vector<double> evd(NUM, 0.0);

  evd = eigenvalues_laplace_dirichlet_2d(NUM);

  for (auto &v: evd)
    std::cout << std::scientific
              << std::showpoint
              << std::setprecision(6)
              <<  v << std::endl;

  //std::vector<double> evn(NUM, 0.0);

  //evn = eigenvalues_laplace_dirichlet_2d(NUM);

  //for (auto &v: evn)
  //  std::cout << v << std::endl;
 
  return 0;
}
