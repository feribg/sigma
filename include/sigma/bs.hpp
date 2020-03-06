#pragma once

#include <cmath>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xindex_view.hpp>
#include "algo.hpp"
#include "instruments.hpp"
#include "smath.hpp"
#include "utils.hpp"

using namespace std;
using namespace algo;
using namespace xt::placeholders;

namespace bs {

struct BSAnalyticResult {
  double npv, delta, gamma, theta, vega;

  [[nodiscard]] std::string toString() const
  {
      return "NPV: "+to_string(npv)+"\n Delta: "+to_string(delta)+"\n Gamma: "+
              to_string(gamma)+"\n Vega:"+to_string(vega)+"\n Theta: "+to_string(theta);
  }
};

struct BSFDResult {
  xt::xtensor<double,2> S, C, delta, gamma, theta, vega;
  [[nodiscard]] std::string toString() const
  {
      return "result";
  }
};

/**
* @param S - spot price
* @param K - strike price
* @param r - annualized interest rate
* @param sigma - annualized volatility
* @param T - time to exp in years
**/
double d1(double S, double K, double r, double sigma, double T)
{
    return (log(S/K)+(r+0.5*gcem::sq(sigma))*T)/(sigma*sqrt(T));
}
/**
* @param S - spot price
* @param K - strike price
* @param r - annualized interest rate
* @param sigma - annualized volatility
* @param T - time to exp in years
**/
double d2(double S, double K, double r, double sigma, double T)
{

    return d1(S, K, r, sigma, T)-sigma*sqrt(T);
}

/**
 * Calculate BS
 * @param S - Stock price
 * @param K - Strike price
 * @param r - Annual rf rate
 * @param sigma - Annual volatility
 * @param div  - Annual div rate
 * @param T - time to expiration in factor of years
 * @param payoff - Type of payoff
 * @return npv,delta,gamma,vega,theta
 */
BSAnalyticResult bs_analytic(double S, double K, double r, double sigma, double div, double T, Payoff payoff)
{
    auto D1 = d1(S, K, r, sigma, T);
    auto D2 = d2(S, K, r, sigma, T);
    auto normcdf_d1 = gcem::normCDF(D1);
    auto normpdf_d1 = gcem::normPDF(D1);
    auto sqrt_t = sqrt(T);
    double div_discount = 1.0;
    if (div!=0) {
        div_discount = exp(-div*T);
    }
    auto rf_discount = exp(-r*T);
    BSAnalyticResult result{};
    result.gamma = div_discount*normpdf_d1/(S*sigma*sqrt_t);  // gamma
    result.vega = div_discount*S*normpdf_d1*sqrt_t; // vega

    if (payoff==CALL) {
        auto normcdf_d2 = gcem::normCDF(D2);
        result.npv = div_discount*normcdf_d1*S-normcdf_d2*K*rf_discount;  //call NPV
        result.delta = div_discount*normcdf_d1;  // delta
        result.theta = -(div_discount*S*normpdf_d1*sigma)/(2*sqrt_t)-r*K*rf_discount*normcdf_d2
                +div*S*div_discount*normcdf_d1;  // theta
    }
    else {
        auto normcdf_minus_d1 = gcem::normCDF(-D1);
        auto normcdf_minus_d2 = gcem::normCDF(-D2);
        result.npv = -div_discount*normcdf_minus_d1*S+normcdf_minus_d2*K*rf_discount;  // put NPV
        result.delta = -div_discount*normcdf_minus_d1;  // delta
        result.theta = -(div_discount*S*normpdf_d1*sigma)/(2*sqrt_t)-r*K*exp(-r*T)*normcdf_minus_d2
                -div*S*div_discount*normcdf_minus_d1;  // theta
    }
    return result;
}

const auto bs_analytic_vectorizer = xt::vectorize(bs_analytic);
/**
 * Batch vectorized implementation of bs_analytic
 * @param S
 * @param K
 * @param r
 * @param sigma
 * @param div
 * @param payoff
 * @param T
 * @return
 */
xt::xtensor<BSAnalyticResult, 1> bs_analytic_batch(
        const xt::xtensor<double, 1>& S,
        const xt::xtensor<double, 1>& K,
        const xt::xtensor<double, 1>& r,
        const xt::xtensor<double, 1>& sigma,
        const xt::xtensor<double, 1>& div,
        const xt::xtensor<double, 1>& T,
        const xt::xtensor<Payoff, 1>& payoff
)
{
    return bs_analytic_vectorizer(S, K, r, sigma, div, T, payoff);
}

BSFDResult fd(long M, double K, double r, double sigma, double div, double T, Payoff payoff, Style style, double rec_dx,
        double rec_dt)
{
    // Step sizes:
    double dt = 0.; // size of time steps
    double dx = 0.; // size of space steps that guarantee explicit stability
    if (rec_dt!=0 && rec_dx!=0) {
        dt = rec_dt;
        dx = rec_dx;
    }else{
        dt = T/((double) M-1);
        dx = sigma*sqrt(dt);
    }

    const double sigma_sq  = gcem::sq(sigma);
    const double alpha = 0.5*sigma_sq;  // diffusivity in heat equation
    const double dn = alpha*dt/(dx*dx);  // diffusion number

    // size of the grid
    //TODO: heuristic for defining those
    const double L1 = -5;
    const double L2 = 1.5;
    const unsigned long N = ceil((L2-L1)/dx+1);

    xt::xtensor<double,2> x = xt::view(xt::linspace<double>(L1, L2, N), xt::newaxis(), xt::all());
    xt::xtensor<double,1> t = xt::linspace<double>(0, T, M);
    xt::xtensor<double,2> tau = xt::expand_dims(t, 1);

    xt::xtensor<double, 2> S = (K*xt::exp(x-(r-div-0.5*sigma_sq)*tau));
    xt::xtensor<double, 2> S_init = xt::view(S, xt::all(), N-1, xt::newaxis());
    // Solution matrix, solve backwards where idx = 0 is exp, idx = N is today
    xt::xtensor<double, 2> U = xt::eval(xt::zeros<double>({M, (long) N}));
    auto U_first_col_v = xt::view(U, xt::all(), 0);
    auto S_first_col_v = xt::view(S, xt::all(), 0);
    auto S_first_row_v = xt::view(S, 0, xt::all());
    auto U_first_row_v = xt::view(U, 0, xt::all());

    // Boundary condition and exercice region
    if (payoff==CALL) {
        auto U_last_col_v = xt::view(U, xt::all(), N-1);
        if (style==EU) {
            U_last_col_v = xt::flatten(xt::exp(r*tau)*xt::maximum(xt::exp(-div*tau)*S_init-K*xt::exp(-r*tau), 0));
        }
        else {
            U_last_col_v = xt::flatten(xt::maximum(xt::maximum(S_init-K, 0)*xt::exp(r*tau),
                    xt::exp(r*tau)*xt::maximum(xt::exp(-div*tau)*S_init-K*xt::exp(-r*tau), 0)));
        }
        U_first_col_v = 0.0;
        U_first_row_v = xt::maximum(S_first_row_v-K, 0);
    }
    else {
        if (style==EU) {
            U_first_col_v = K-S_first_col_v*xt::exp(r*tau); // Merton Bound
        }
        else {
            U_first_col_v = (K-S_first_col_v)*xt::exp(r*tau);  // early exercise-need to double check for Calls!
        }
        auto U_last_col_v = xt::view(U, xt::all(), N-1);
        U_last_col_v = 0;
        U_first_row_v = xt::maximum(K-S_first_row_v, 0);
    }

    // Tridiagonal scheme equations
    const double theta_ = 1./2.;  //TODO: may want to alter this through time, in which case just place in loop below
    xt::xtensor<double, 2> ones_tmp = xt::ones<double>({(long) N-2, 1L});
    xt::xtensor<double, 1> main_diag = xt::flatten((1.+2.*theta_*dn)*ones_tmp);
    xt::xtensor<double, 1> off_diag = xt::flatten(-theta_*dn*ones_tmp);

    for (long i = 1; i<M; i++) {
        xt::xtensor<double, 1> d = xt::xtensor<double, 1>::from_shape({N-2});
        for (long j = 1; j<N-1; j++) {
            d(j-1) = (1-theta_)*dn*U(i-1, j+1)+(1-2*(1-theta_)*dn)*U(i-1, j)+(
                    1-theta_)*dn*U(i-1, j-1);
        }
        d(0) = d(0)+dn*theta_*U(i, 0); // from boundary conditions
        d(N-3) = d(N-3)+dn*theta_*U(i, N-1);

        if (style==EU) {
            xt::xtensor<double, 1> U_tmp_v = xt::view(U, i, xt::range(1, N));
            algo::thomas_tridiag(main_diag, off_diag, off_diag, d, U_tmp_v);
        }
        else {
            // Solve for american in explicit SOR
            double err = std::numeric_limits<double>::infinity();
            const double cutoff = gcem::pow(10.0, -3);

            xt::xtensor<double, 1> U_initial = xt::eval(d);
            double w = 1.5; // step size
            auto S_current_v = xt::view(S, i, xt::range(1, N-1));
            xt::xtensor<double, 1> U_next;
            if (payoff==CALL) {
                while(err > cutoff){
                    xt::xtensor<double,1> step_result = U_initial;
                    //TODO: look into minimizing copies here
                    algo::gs_tridiag_step(U_initial, main_diag, off_diag, off_diag, d, N-2, step_result);
                    auto lhs = (1-w)*U_initial+w*step_result;
                    auto rhs = exp(r*tau(i,0))*(S_current_v-K);
                    U_next = xt::maximum(lhs,rhs);
                    err = xt::linalg::norm(U_next - U_initial, xt::linalg::normorder::inf);
                    U_initial = U_next;
                    //TODO: add convergence test
                }
            }
            else {
                while(err > cutoff){
                    //TODO: look into minimizing copies here
                    auto step_result = U_initial;
                    algo::gs_tridiag_step(U_initial, main_diag, off_diag, off_diag, d, N-2, step_result);
                    U_next = xt::amax(xt::hstack(xt::xtuple(
                            (1-w)*U_initial+w*step_result,
                            exp(r*tau(i))*(K-S_current_v)
                            )));
                    err = xt::linalg::norm(U_next - U_initial, xt::linalg::normorder::inf);
                    U_initial = U_next;
                    //TODO: add convergence test
                }
            }
            auto U_current_row_v = xt::view(U, i, xt::range(1, N-1));
            U_current_row_v = U_next;
        }
    }
    // put back into C out of U
    xt::xtensor<double, 2> ones_N = xt::ones<double>({1L, (long)N});
    xt::xtensor<double, 2> C = xt::exp(-r * tau * ones_N) * U;

    // Greeks: chain rule with (x, tau) cordinates, central diff for Delta and Gamma
    // delta
    auto dc_dx_v1 = xt::view(C, xt::all(), xt::range(2, _));
    auto dc_dx_v2 = xt::view(C, xt::all(), xt::range(0,N-2));
    xt::xtensor<double, 2> dc_dx = (dc_dx_v1 - dc_dx_v2) / (2 * dx);
    //TODO: avoid aliasing extra copy by folding it above

    auto dc_dx_v = xt::concatenate(xtuple(xt::view(dc_dx, xt::all(), 0, xt::newaxis()), dc_dx, xt::view(dc_dx, xt::all(), N-3, xt::newaxis())), 1);
    xt::xtensor<double,2> delta = dc_dx_v / S;
    // gamma
    xt::xtensor<double, 2> dc2_dx2 = xt::diff(C, 2, 1) / (dx * dx);
    auto dc2_dx2_v = xt::concatenate(xtuple(xt::view(dc2_dx2, xt::all(), 0, xt::newaxis()), dc2_dx2, xt::view(dc2_dx2, xt::all(), N-3, xt::newaxis())), 1);
    auto chain = dc2_dx2_v - dc_dx_v;
    xt::xtensor<double, 2> gamma = chain / (S * S);
    //TODO: why hardcode the threshold to 2
    filtration(gamma, S < 2.0) = 0.;  // fix boundary errors

    // Numerical theta, Forward difference for Thetas
    xt::xtensor<double,2> zeros = xt::zeros<double>({1L, (long)N});

    xt::xtensor<double, 2> dc_dtau = xt::concatenate(xtuple(zeros, xt::diff(C, 1, 0)), 0) /  dt;  //TODO: stability check here for dt ~ 0 ?
    xt::xtensor<double, 2> theta = - dc_dx_v * (r - div - 0.5 * sigma_sq) - dc_dtau;
    xt::view(theta, 0, xt::all()) = 0;
    // Analytic theta
    //theta = -0.5 * (sigma_sq) * (S * S) * gamma  // look into effect of IR/Div

    // Numeric vega
    xt::xtensor<double, 2> vega;
    if(style == EU && (rec_dt==0 && rec_dx==0)){
        vega = (sigma * tau * ones_N) * (S * S * gamma);
    }else if(rec_dt==0 && rec_dx==0){
        auto dvol = 0.05; //TODO: what's the correct value here should be % rather than absolute
        auto res1 = bs::fd(M, K, r, sigma+dvol, div, T, payoff, style, dx, dt);
        auto res2 = bs::fd(M, K, r, sigma-dvol, div, T, payoff, style, dx, dt);
        vega = (res1.C - res2.C - delta * (res1.S - res2.S) - 0.5 * gamma * (gcem::sq(res1.S - S) - gcem::sq(res2.S - S))) / (2 * dvol);

        //faster(only one more loop) but less accurate
        //auto res1 = fd(M, K, r, div, T, sigma + dvol, style, payoff, (dx, dt))
        //vega = (res1.C - C - delta * (res1.S - S) - 0.5 * gamma * ((res1.S - S) ** 2)) / dvol
    }
    return BSFDResult{
        .S=std::move(S),
        .C=std::move(C),
        .delta=std::move(delta),
        .gamma=std::move(gamma),
        .theta=std::move(theta),
        .vega=std::move(vega),

    };
}

}