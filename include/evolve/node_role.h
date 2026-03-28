#pragma once
#include <cstdint>

namespace evolve {

enum class NodeRole : uint8_t {
    Input  = 0,
    Hidden = 1,
    Output = 2,
};

} // namespace evolve
