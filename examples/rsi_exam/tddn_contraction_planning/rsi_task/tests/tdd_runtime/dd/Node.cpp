#include "Node.hpp"

namespace dd {


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
mNode mNode::terminal{{{{nullptr, Complex::zero},
                        {nullptr, Complex::zero}}},
                      nullptr,
                      0U,
                      -1,
                      32 + 16};


} // namespace dd
