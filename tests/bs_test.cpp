#include <iostream>
#include "sigma/bs.hpp"
#include "chrono"
#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"

using namespace std::chrono;

namespace {

TEST(BS_Analytic, CallWithDivAndRf)
{

    double K = 100.0; // strike
    double r = 0.0194;  // LIBOR IR
    double d = 0.017;  // SPY div
    double T = 1;  // 1 year
    double sigma = 0.2;  // 20% annual vol
    Payoff payoff = CALL;
    double S = 120.0;

    auto res = bs::bs_analytic(S, K, r, sigma, d, T, payoff);
    EXPECT_DOUBLE_EQ(21.941832577343249, res.npv);
    EXPECT_DOUBLE_EQ(0.85159934538321957, res.delta);
    EXPECT_DOUBLE_EQ(0.008839738600121395, res.gamma);
    EXPECT_DOUBLE_EQ(-2.3654337763048705, res.theta);
    EXPECT_DOUBLE_EQ(25.458447168349622, res.vega);
}

TEST(BS_Analytic, CallWithDivAndRf_Batch)
{

    xt::xarray<double> K = {100.0, 100.0, 100.0}; // strike
    xt::xarray<double> r = {0.0194, 0.0194, 0.0194};  // LIBOR IR
    xt::xarray<double> d = {0.017, 0.017, 0.017};  // SPY div
    xt::xarray<double> T = {1., 1., 1.};  // 1 year
    xt::xarray<double> sigma = {0.2, 0.2, 0.2};  // 20% annual vol
    xt::xarray<Payoff> payoff = {CALL, CALL, CALL};
    xt::xarray<double> S = {120.0, 120.0, 120.0};

    auto res = bs::bs_analytic_batch(S, K, r, sigma, d, T, payoff);
    EXPECT_DOUBLE_EQ(21.941832577343249, res(0).npv);
    EXPECT_DOUBLE_EQ(res(1).npv, res(0).npv);
    EXPECT_DOUBLE_EQ(res(2).npv, res(1).npv);
    EXPECT_DOUBLE_EQ(0.85159934538321957, res(0).delta);
    EXPECT_DOUBLE_EQ(res(1).delta, res(0).delta);
    EXPECT_DOUBLE_EQ(res(2).delta, res(1).delta);
    EXPECT_DOUBLE_EQ(0.008839738600121395, res(0).gamma);
    EXPECT_DOUBLE_EQ(res(1).gamma, res(0).gamma);
    EXPECT_DOUBLE_EQ(res(2).gamma, res(1).gamma);
    EXPECT_DOUBLE_EQ(-2.3654337763048705, res(0).theta);
    EXPECT_DOUBLE_EQ(res(1).theta, res(0).theta);
    EXPECT_DOUBLE_EQ(res(2).theta, res(1).theta);
    EXPECT_DOUBLE_EQ(25.458447168349622, res(0).vega);
    EXPECT_DOUBLE_EQ(res(1).vega, res(0).vega);
    EXPECT_DOUBLE_EQ(res(2).vega, res(1).vega);
}

TEST(BS_FD, CallAmWithDivAndRf)
{

    double K = 100.0; // strike
    double r = 0.0194;  // LIBOR IR
    double d = 0.017;  // SPY div
    double T = 1;  // 1 year
    double sigma = 0.2;  // 20% annual vol
    auto M = 3 * 365; // grid steps
    Payoff payoff = CALL;
    Style style = AM;

    auto res = bs::fd(M, K, r, sigma, d, T, payoff, style, 0, 0);

    auto shape = res.S.shape();
    auto rix = shape[0];
    auto cix = shape[1];
    EXPECT_DOUBLE_EQ(0.67379469990854668, res.S(0,0));
    EXPECT_DOUBLE_EQ(0.715907850860794, res.S(10,10));
    EXPECT_DOUBLE_EQ(0.685758458883008, res.S(rix-1,0));
    EXPECT_DOUBLE_EQ(4.481689070338065e+02, res.S(0,cix-1));
    EXPECT_DOUBLE_EQ(4.561265012154291e+02, res.S(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(4.319054901783607e+02, res.S(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.C(0,0));
    EXPECT_DOUBLE_EQ(0., res.C(10,10));
    EXPECT_DOUBLE_EQ(0., res.C(rix-1,0));
    EXPECT_DOUBLE_EQ(3.481689070338065e+02, res.C(0,cix-1));
    EXPECT_DOUBLE_EQ(3.561265012154291e+02, res.C(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(3.319054901783607e+02, res.C(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.delta(0,0));
    EXPECT_DOUBLE_EQ(0., res.delta(10,10));
    EXPECT_DOUBLE_EQ(1.7001614900423999e-129, res.delta(rix-1,0));
    EXPECT_DOUBLE_EQ(0.993940157374373, res.delta(0,cix-1));
    EXPECT_DOUBLE_EQ(0.99394015737439212, res.delta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.99996823414579061, res.delta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.gamma(0,0));
    EXPECT_DOUBLE_EQ(0., res.gamma(10,10));
    EXPECT_DOUBLE_EQ(0., res.gamma(rix-1,0));
    EXPECT_DOUBLE_EQ(-9.0719544277341756e-08, res.gamma(0,cix-1));
    EXPECT_DOUBLE_EQ(-8.9136842273589639e-08, res.gamma(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-9.4706563941069902e-08, res.gamma(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.theta(0,0));
    EXPECT_DOUBLE_EQ(0., res.theta(10,10));
    EXPECT_DOUBLE_EQ(2.0519842169441393e-131, res.theta(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.theta(0,cix-1));
    EXPECT_DOUBLE_EQ(-0.048582790254290131, res.theta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-0.00018032376364196523, res.theta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.vega(0,0));
    EXPECT_DOUBLE_EQ(0., res.vega(10,10));
    EXPECT_DOUBLE_EQ(-2.3347557314554352e-130, res.vega(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.vega(0,cix-1));
    EXPECT_DOUBLE_EQ(0.5535164647594808, res.vega(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.0027293413578043961, res.vega(rix-10,cix-10));
}

TEST(BS_FD, PutAmWithDivAndRf)
{

    double K = 100.0; // strike
    double r = 0.0194;  // LIBOR IR
    double d = 0.017;  // SPY div
    double T = 1;  // 1 year
    double sigma = 0.2;  // 20% annual vol
    auto M = 3 * 365; // grid steps
    Payoff payoff = CALL;
    Style style = AM;

    auto res = bs::fd(M, K, r, sigma, d, T, payoff, style, 0, 0);

    auto shape = res.S.shape();
    auto rix = shape[0];
    auto cix = shape[1];
    EXPECT_DOUBLE_EQ(0.67379469990854668, res.S(0,0));
    EXPECT_DOUBLE_EQ(0.715907850860794, res.S(10,10));
    EXPECT_DOUBLE_EQ(0.685758458883008, res.S(rix-1,0));
    EXPECT_DOUBLE_EQ(4.481689070338065e+02, res.S(0,cix-1));
    EXPECT_DOUBLE_EQ(4.561265012154291e+02, res.S(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(4.319054901783607e+02, res.S(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.C(0,0));
    EXPECT_DOUBLE_EQ(0., res.C(10,10));
    EXPECT_DOUBLE_EQ(0., res.C(rix-1,0));
    EXPECT_DOUBLE_EQ(3.481689070338065e+02, res.C(0,cix-1));
    EXPECT_DOUBLE_EQ(3.561265012154291e+02, res.C(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(3.319054901783607e+02, res.C(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.delta(0,0));
    EXPECT_DOUBLE_EQ(0., res.delta(10,10));
    EXPECT_DOUBLE_EQ(1.7001614900423999e-129, res.delta(rix-1,0));
    EXPECT_DOUBLE_EQ(0.993940157374373, res.delta(0,cix-1));
    EXPECT_DOUBLE_EQ(0.99394015737439212, res.delta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.99996823414579061, res.delta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.gamma(0,0));
    EXPECT_DOUBLE_EQ(0., res.gamma(10,10));
    EXPECT_DOUBLE_EQ(0., res.gamma(rix-1,0));
    EXPECT_DOUBLE_EQ(-9.0719544277341756e-08, res.gamma(0,cix-1));
    EXPECT_DOUBLE_EQ(-8.9136842273589639e-08, res.gamma(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-9.4706563941069902e-08, res.gamma(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.theta(0,0));
    EXPECT_DOUBLE_EQ(0., res.theta(10,10));
    EXPECT_DOUBLE_EQ(2.0519842169441393e-131, res.theta(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.theta(0,cix-1));
    EXPECT_DOUBLE_EQ(-0.048582790254290131, res.theta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-0.00018032376364196523, res.theta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.vega(0,0));
    EXPECT_DOUBLE_EQ(0., res.vega(10,10));
    EXPECT_DOUBLE_EQ(-2.3347557314554352e-130, res.vega(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.vega(0,cix-1));
    EXPECT_DOUBLE_EQ(0.5535164647594808, res.vega(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.0027293413578043961, res.vega(rix-10,cix-10));
}

TEST(BS_FD, CallEuWithDivAndRf)
{

    double K = 100.0; // strike
    double r = 0.0194;  // LIBOR IR
    double d = 0.017;  // SPY div
    double T = 1;  // 1 year
    double sigma = 0.2;  // 20% annual vol
    auto M = 3 * 365; // grid steps
    Payoff payoff = CALL;
    Style style = AM;

    auto res = bs::fd(M, K, r, sigma, d, T, payoff, style, 0, 0);

    auto shape = res.S.shape();
    auto rix = shape[0];
    auto cix = shape[1];
    EXPECT_DOUBLE_EQ(0.67379469990854668, res.S(0,0));
    EXPECT_DOUBLE_EQ(0.715907850860794, res.S(10,10));
    EXPECT_DOUBLE_EQ(0.685758458883008, res.S(rix-1,0));
    EXPECT_DOUBLE_EQ(4.481689070338065e+02, res.S(0,cix-1));
    EXPECT_DOUBLE_EQ(4.561265012154291e+02, res.S(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(4.319054901783607e+02, res.S(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.C(0,0));
    EXPECT_DOUBLE_EQ(0., res.C(10,10));
    EXPECT_DOUBLE_EQ(0., res.C(rix-1,0));
    EXPECT_DOUBLE_EQ(3.481689070338065e+02, res.C(0,cix-1));
    EXPECT_DOUBLE_EQ(3.561265012154291e+02, res.C(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(3.319054901783607e+02, res.C(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.delta(0,0));
    EXPECT_DOUBLE_EQ(0., res.delta(10,10));
    EXPECT_DOUBLE_EQ(1.7001614900423999e-129, res.delta(rix-1,0));
    EXPECT_DOUBLE_EQ(0.993940157374373, res.delta(0,cix-1));
    EXPECT_DOUBLE_EQ(0.99394015737439212, res.delta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.99996823414579061, res.delta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.gamma(0,0));
    EXPECT_DOUBLE_EQ(0., res.gamma(10,10));
    EXPECT_DOUBLE_EQ(0., res.gamma(rix-1,0));
    EXPECT_DOUBLE_EQ(-9.0719544277341756e-08, res.gamma(0,cix-1));
    EXPECT_DOUBLE_EQ(-8.9136842273589639e-08, res.gamma(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-9.4706563941069902e-08, res.gamma(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.theta(0,0));
    EXPECT_DOUBLE_EQ(0., res.theta(10,10));
    EXPECT_DOUBLE_EQ(2.0519842169441393e-131, res.theta(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.theta(0,cix-1));
    EXPECT_DOUBLE_EQ(-0.048582790254290131, res.theta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-0.00018032376364196523, res.theta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.vega(0,0));
    EXPECT_DOUBLE_EQ(0., res.vega(10,10));
    EXPECT_DOUBLE_EQ(-2.3347557314554352e-130, res.vega(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.vega(0,cix-1));
    EXPECT_DOUBLE_EQ(0.5535164647594808, res.vega(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.0027293413578043961, res.vega(rix-10,cix-10));
}

TEST(BS_FD, PutEuWithDivAndRf)
{

    double K = 100.0; // strike
    double r = 0.0194;  // LIBOR IR
    double d = 0.017;  // SPY div
    double T = 1;  // 1 year
    double sigma = 0.2;  // 20% annual vol
    auto M = 3 * 365; // grid steps
    Payoff payoff = CALL;
    Style style = AM;

    auto res = bs::fd(M, K, r, sigma, d, T, payoff, style, 0, 0);

    auto shape = res.S.shape();
    auto rix = shape[0];
    auto cix = shape[1];
    EXPECT_DOUBLE_EQ(0.67379469990854668, res.S(0,0));
    EXPECT_DOUBLE_EQ(0.715907850860794, res.S(10,10));
    EXPECT_DOUBLE_EQ(0.685758458883008, res.S(rix-1,0));
    EXPECT_DOUBLE_EQ(4.481689070338065e+02, res.S(0,cix-1));
    EXPECT_DOUBLE_EQ(4.561265012154291e+02, res.S(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(4.319054901783607e+02, res.S(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.C(0,0));
    EXPECT_DOUBLE_EQ(0., res.C(10,10));
    EXPECT_DOUBLE_EQ(0., res.C(rix-1,0));
    EXPECT_DOUBLE_EQ(3.481689070338065e+02, res.C(0,cix-1));
    EXPECT_DOUBLE_EQ(3.561265012154291e+02, res.C(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(3.319054901783607e+02, res.C(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.delta(0,0));
    EXPECT_DOUBLE_EQ(0., res.delta(10,10));
    EXPECT_DOUBLE_EQ(1.7001614900423999e-129, res.delta(rix-1,0));
    EXPECT_DOUBLE_EQ(0.993940157374373, res.delta(0,cix-1));
    EXPECT_DOUBLE_EQ(0.99394015737439212, res.delta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.99996823414579061, res.delta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.gamma(0,0));
    EXPECT_DOUBLE_EQ(0., res.gamma(10,10));
    EXPECT_DOUBLE_EQ(0., res.gamma(rix-1,0));
    EXPECT_DOUBLE_EQ(-9.0719544277341756e-08, res.gamma(0,cix-1));
    EXPECT_DOUBLE_EQ(-8.9136842273589639e-08, res.gamma(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-9.4706563941069902e-08, res.gamma(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.theta(0,0));
    EXPECT_DOUBLE_EQ(0., res.theta(10,10));
    EXPECT_DOUBLE_EQ(2.0519842169441393e-131, res.theta(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.theta(0,cix-1));
    EXPECT_DOUBLE_EQ(-0.048582790254290131, res.theta(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(-0.00018032376364196523, res.theta(rix-10,cix-10));

    EXPECT_DOUBLE_EQ(0., res.vega(0,0));
    EXPECT_DOUBLE_EQ(0., res.vega(10,10));
    EXPECT_DOUBLE_EQ(-2.3347557314554352e-130, res.vega(rix-1,0));
    EXPECT_DOUBLE_EQ(0., res.vega(0,cix-1));
    EXPECT_DOUBLE_EQ(0.5535164647594808, res.vega(rix-1,cix-1));
    EXPECT_DOUBLE_EQ(0.0027293413578043961, res.vega(rix-10,cix-10));
}

}
