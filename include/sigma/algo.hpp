#pragma once
#include "utils.hpp"

namespace algo {

// Vectors a, b, c and d are const. They will not be modified
// by the function. Vector f (the solution vector) is non-const
// and thus will be calculated and updated by the function.
// https://www.quantstart.com/articles/Tridiagonal-Matrix-Algorithm-Thomas-Algorithm-in-C/
static inline void thomas_tridiag(
        const xt::xtensor<double, 1>& a,
        const xt::xtensor<double, 1>& b,
        const xt::xtensor<double, 1>& c,
        const xt::xtensor<double, 1>& d,
        xt::xtensor<double, 1>& f)
{
    size_t N = d.shape()[0];

    // TODO: pass temporaries as arguments
    // Note that this is inefficient as it is possible to call
    // this function many times. A better implementation would
    // pass these temporary matrices by non-const reference to
    // save excess allocation and deallocation
    auto c_star = xt::xtensor<double, 1>::from_shape({N});
    auto d_star = xt::xtensor<double, 1>::from_shape({N});

    // This updates the coefficients in the first row
    // Note that we should be checking for division by zero here
    c_star(0) = c(0)/b(0);
    d_star(0) = d(0)/b(0);

    // Create the c_star and d_star coefficients in the forward sweep
    // TODO: simd ?
    for (int i = 1; i<N; i++) {
        double m = (b(i)-a(i)*c_star(i-1));
        c_star(i) = c(i)/m;
        d_star(i) = (d(i)-a(i)*d_star(i-1))/m;
    }

    // This is the reverse sweep, used to update the solution vector f
    for (auto i = N-1; i-->0;) {
        f(i) = d_star(i)-c_star(i)*d(i+1);
    }
}

//TODO: vectorize if possible
/**
 * Naive grid step implementation for the path dependent case
 * @param x
 * @param a
 * @param b
 * @param c
 * @param d
 * @param n
 * @param x_result
 */
static inline void gs_tridiag_step(
        const xt::xtensor<double,1>& x,
        const xt::xtensor<double,1>& a,
        const xt::xtensor<double,1>& b,
        const xt::xtensor<double,1>& c,
        const xt::xtensor<double,1>& d,
        unsigned long n,
        xt::xtensor<double, 1>& x_result)
{
    n = n-1; //TODO do we need that
    x_result(0) = (d(0)-b(0)*x_result(1))/a(0);
    for (auto i = 1; i<n; i++) {
        x_result(i) = (d(i)-c(i-1)*x_result(i-1)-b(i)*x_result(i+1))/a(i);
    }
    x_result(n) = (d(n)-c(n-1)*x_result(n-1))/a(n);
}

}

