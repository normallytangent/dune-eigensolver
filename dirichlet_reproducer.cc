// C++ includes
#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>

#define SIZE 4

#ifndef NUM
#define NUM SIZE
#endif

std::vector<double> eigenvalues_laplace_dirichlet_2d(int N, std::vector<double> &ev_lambda)
{
  std::vector<double> eval(N * N);
  ev_lambda.resize(N * N);
  double h = 1 / (N + 1.0);
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      eval[j * N + i] = 4.0 * (std::sin(0.5 * h * (i + 1) * M_PI) * std::sin(0.5 * h * (i + 1) * M_PI) + std::sin(0.5 * h * (j + 1) * M_PI) * std::sin(0.5 * h * (j + 1) * M_PI));
      ev_lambda[j * N + i] = eval [j*N + i] * M_PI * M_PI * (0.5 * 0.5 * h * h + 0.5 * 0.5 * h * h);
}
}

  std::sort(eval.begin(), eval.end(), std::greater{});
  std::sort(ev_lambda.begin(), ev_lambda.end(), std::greater{});
  return eval;
}

std::vector<double> eigenvalues_laplace_neumann_2d(int N)
{
  std::vector<double> eval(N * N);
  double h = 1 / (N + 1.0);
  
  // https://youtu.be/ik2_5QVVLLA?si=IDc37pJrnrah7PKC
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      eval[j * N + i] = 4.0 * (std::cos(0.5 * h * (i + 1) * M_PI) * std::cos(0.5 * h * (i + 1) * M_PI) + std::cos(0.5 * h * (j + 1) * M_PI) * std::cos(0.5 * h * (j + 1) * M_PI));
}
}
//      ev[j * N + i] = 8.0 * (std::cosh(h * (2j - 1)) * std::cosh(h * (2i - 1)) / std::sinh(h * (j + 1) * M_PI) * std::cosh(h * (j + 1) * M_PI));
  
 // https://math.stackexchange.com/a/802003/288799 
 // for (std::size_t i = 0; i < N; ++i)
 //   for (std::size_t j = 0; j < N; ++j)
 //     ev[j * N + i] = 4.0 * (std::cosh(h * (i + 1) * M_PI) * std::cosh(h * (i + 1) * M_PI) + std::cosh(h * (j + 1) * M_PI) * std::cosh(h * (j + 1) * M_PI));

  std::sort(eval.begin(), eval.end(), std::greater{});
  return eval;
}

int main() {
  
  std::vector<double> evd(NUM, 0.0);
  std::vector<double> evd_lambda(NUM, 0.0);

  evd = eigenvalues_laplace_dirichlet_2d(NUM, evd_lambda);

  for (auto &v: evd) {
    std::cout << std::scientific
              << std::showpoint
              << std::setprecision(6)
              <<  v << '\n';
}
  std::cout << '\n';
  for (auto &v: evd_lambda) {
    std::cout << std::scientific
              << std::showpoint
              << std::setprecision(6)
              <<  v << '\n';
}

  // std::vector<double> evn(NUM, 0.0);

//  evn = eigenvalues_laplace_neumann_2d(NUM);
//   std::cout << "Neumann Problem: \n";

//   for (auto &w: evn) {
//     std::cout << std::scientific
//               << std::showpoint
//               << std::setprecision(6)
//               << w << '\n';
// }
 
  return 0;
}
