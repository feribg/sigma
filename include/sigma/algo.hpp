#pragma once
#include "utils.hpp"
#include "smath.hpp"
#include "xtensor/xtensor.hpp"

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
 * Gauss Siedel tridiagonal solver
 * @param x_end
 * @param x_start
 * @param bound
 * @param a
 * @param b
 * @param c
 * @param d
 * @param n
 * @param w
 * @return
 */
template <class V1, class V2, class V3>
static inline double gs_tridiag_step(
        V1& x_end,
        const V2& bound,
        const xt::xtensor<double,1>& a,
        const xt::xtensor<double,1>& b,
        const xt::xtensor<double,1>& c,
        const V3& d,
        unsigned long n,
        double w)
{
    double err = 0.0;
    auto init = x_end(0);
    auto next_gs = (d(0)-b(0)*x_end(1))/a(0);
    auto next_bound = std::max((1-w)*init+w*next_gs, bound(0));
    x_end(0) = next_bound;
    err = std::max(err, std::abs(next_bound - init));
    for (auto i = 1; i<n; i++) {
        init = x_end(i);
        next_gs = (d(i)-c(i-1)*next_gs-b(i)*x_end(i+1))/a(i);
        next_bound = std::max((1-w)*init+w*next_gs, bound(i));
        x_end(i) = next_bound;
        err = std::max(err, std::abs(next_bound - init));
    }
    init = x_end(n);
    next_gs = (d(n)-c(n-1)*next_gs)/a(n);
    next_bound = std::max((1-w)*init+w*next_gs, bound(n));
    x_end(n) = next_bound;
    err = std::max(err, std::abs(next_bound - init));
    return err; //inf norm
}

}

