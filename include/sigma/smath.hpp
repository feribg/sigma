#pragma once

#include <cmath>
#include "gcem.hpp"
/**
 * Compile time math expressions extending the provided ones by gcem library
 */
namespace gcem {
/**
 * Helper methods
 */
namespace detail {
/**
 * Consts
 */
const double ONE_O_SQRT_2PI = 0.39894228040143267793994605993438;
const double SQRT_1_O_2 = sqrt(0.5);
}

/**
 * Square
 * @tparam Number
 * @param x
 * @return
 */
template<typename Number>
constexpr auto sq(Number x)
-> decltype(auto)
{
    return x*x;
}

/**
 * Norm PDF function for x
 * @tparam Number
 * @param x
 * @return
 */
template<typename Number>
constexpr auto normPDF(Number x)
{
    return detail::ONE_O_SQRT_2PI*exp(-0.5*sq(x));
}

/**
 * Norm CDF function for X
 * @tparam Number
 * @param x
 * @return
 */
template<typename Number>
constexpr auto normCDF(Number x)
{
    return 0.5*erfc(-x*detail::SQRT_1_O_2);
}

}