#pragma once
#include <evolve/node_role.h>
#include <cstdint>
#include <vector>

namespace evolve {

struct EmptyProps {};

template <typename Props = EmptyProps>
struct NodeGene {
    uint32_t id;
    NodeRole role;
    [[no_unique_address]] Props props;
};

struct ConnectionGene {
    uint32_t from_node;
    uint32_t to_node;
    float weight;
    bool enabled;
    uint32_t innovation;
};

template <typename Props = EmptyProps>
struct GraphGenome {
    std::vector<NodeGene<Props>> nodes;
    std::vector<ConnectionGene> connections;
};

} // namespace evolve
