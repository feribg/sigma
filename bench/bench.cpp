#include <benchmark/benchmark.h>
#include "sigma/bs.hpp"
#include "xtensor/xadapt.hpp"

static void BM_bs_analytic(benchmark::State& state)
{
    size_t n = 100;
    std::vector<double> K(n, 100.0); // strike
    std::vector<double> r(n, 0.0194);  // LIBOR IR
    std::vector<double> d(n, 0.017);  // SPY div
    std::vector<double> T(n, 1);  // 1 year
    std::vector<double> sigma(n, 0.2);  // 20% annual vol
    std::vector<double> M(n, 3*365);  // number of time steps
    std::vector<Payoff> payoff(n, CALL);
    std::vector<Style> style(n, AM);
    std::vector<double> S(n, 120.0);
    auto res = xt::xtensor<bs::BSAnalyticResult, 1>::from_shape({1000});

    for (auto _ : state) {
        for (size_t i = 0; i<res.size(); ++i) {
            res[i] = bs::bs_analytic(S[i], K[i], r[i], sigma[i], d[i], T[i], payoff[i]);
        }
    }
}
BENCHMARK(BM_bs_analytic)->Iterations(100)->Repetitions(100)->ReportAggregatesOnly(true)->DisplayAggregatesOnly(true);

static void BM_bs_analytic_batch(benchmark::State& state)
{
    size_t n = 100;
    xt::xtensor<double, 1> K = xt::adapt(std::vector<double>(n, 100.0));
    xt::xtensor<double, 1> r = xt::adapt(std::vector<double>(n, 0.0194));
    xt::xtensor<double, 1> d = xt::adapt(std::vector<double>(n, 0.017));
    xt::xtensor<double, 1> T = xt::adapt(std::vector<double>(n, 1.0));
    xt::xtensor<double, 1> sigma = xt::adapt(std::vector<double>(n, 0.2));
    xt::xtensor<Payoff, 1> payoff = xt::adapt(std::vector<Payoff>(n, Payoff::CALL));
    xt::xtensor<double, 1> S = xt::adapt(std::vector<double>(n, 120.0));
    auto res = xt::xtensor<bs::BSAnalyticResult, 1>::from_shape({1000});

    for (auto _ : state) {
        xt::noalias(res) = bs::bs_analytic_batch(S, K, r, sigma, d, T, payoff);
    }
}
BENCHMARK(BM_bs_analytic_batch)->Iterations(100)->Repetitions(100)->ReportAggregatesOnly(true)->DisplayAggregatesOnly(
        true);

static void BM_bs_fd(benchmark::State& state)
{
    double K = 100.0; // strike
    double r = 0.0194;  // LIBOR IR
    double d = 0.017;  // SPY div
    double T = 1;  // 1 year
    double sigma = 0.2;  // 20% annual vol
    auto M = 3*365; // grid steps
    Payoff payoff = CALL;
    Style style = AM;

    for (auto _ : state) {
        auto res = bs::fd(M, K, r, sigma, d, T, payoff, style, 0, 0);
    }
}
BENCHMARK(BM_bs_fd)->Iterations(5)->Repetitions(1)->ReportAggregatesOnly(true)->DisplayAggregatesOnly(true);

BENCHMARK_MAIN();