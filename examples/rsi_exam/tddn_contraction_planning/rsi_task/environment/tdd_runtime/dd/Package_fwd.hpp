#pragma once
#include "DDpackageConfig.hpp"
#include "Node.hpp"

namespace qc {

using MatrixDD = dd::mEdge;

} // namespace qc

namespace dd {
template <class Config = DDPackageConfig> class Package;
} // namespace dd
