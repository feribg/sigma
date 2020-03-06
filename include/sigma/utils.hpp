#pragma once

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>


#define XT_DBG(xexpr)  do { namespace po = xt::print_options; \
std::cout << std::endl << #xexpr << "Shape: " << xt::adapt(xexpr.shape()) << std::endl; \
std::cout << "==================================================================================" << std::endl; \
std::cout << po::line_width(500) << po::threshold(100) << po::precision(10) << po::edge_items(20) << xexpr << std::endl; \
std::cout << "==================================================================================" << std::endl; \
} while (0)
