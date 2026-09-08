#pragma once


#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include "Tdd.hpp"

namespace dd {

    struct Tensor {

        xt::xarray<ComplexValue> data;

        std::vector<Index> index_set;

        std::string name=NULL;

    };


    struct TensorNetwork {
        std::vector<Tensor> tensors;
    };


}