#pragma once
#include <evolve/graph_gene.h>
#include <functional>
#include <random>
#include <stdexcept>

namespace evolve {

template <typename Props>
struct NeatPolicy {
    std::function<void(Props& props, std::mt19937& rng)> init_node_props;

    std::function<void(Props& child_props,
                       const Props& parent_a,
                       const Props& parent_b,
                       std::mt19937& rng)> merge_node_props;

    std::function<void(GraphGenome<Props>& genome, std::mt19937& rng)> mutate_properties;

    std::function<void(Props& props, std::mt19937& rng)> init_output_node_props;

    void validate() const {
        if (!init_node_props)
            throw std::invalid_argument("NeatPolicy: init_node_props callback is null");
        if (!merge_node_props)
            throw std::invalid_argument("NeatPolicy: merge_node_props callback is null");
        if (!mutate_properties)
            throw std::invalid_argument("NeatPolicy: mutate_properties callback is null");
        if (!init_output_node_props)
            throw std::invalid_argument("NeatPolicy: init_output_node_props callback is null");
    }
};

inline NeatPolicy<EmptyProps> default_neat_policy() {
    return {
        .init_node_props      = [](EmptyProps&, std::mt19937&) {},
        .merge_node_props     = [](EmptyProps&, const EmptyProps&, const EmptyProps&, std::mt19937&) {},
        .mutate_properties    = [](GraphGenome<>&, std::mt19937&) {},
        .init_output_node_props = [](EmptyProps&, std::mt19937&) {},
    };
}

} // namespace evolve
