#pragma once
#include <evolve/graph_gene.h>
#include <evolve/innovation.h>
#include <evolve/neat_policy.h>
#include <random>

namespace evolve {

struct NeatWeightConfig {
    float weight_mutate_rate      = 0.80f;
    float weight_perturb_rate     = 0.90f;
    float weight_perturb_strength = 0.3f;
    float weight_replace_range    = 2.0f;
};

struct NeatMutationConfig {
    NeatWeightConfig weights;
    float add_connection_rate     = 0.10f;
    float add_node_rate           = 0.03f;
    float disable_connection_rate = 0.02f;
};

struct SpeciationConfig {
    float compatibility_threshold = 3.0f;
    float c_excess    = 1.0f;
    float c_disjoint  = 1.0f;
    float c_weight    = 0.4f;
};

// --- Structural mutations ---

template <typename Props>
void mutate_weights(GraphGenome<Props>& genome, const NeatWeightConfig& config, std::mt19937& rng);

template <typename Props>
void add_connection(GraphGenome<Props>& genome, InnovationCounter& counter, std::mt19937& rng);

template <typename Props>
void add_node(GraphGenome<Props>& genome, InnovationCounter& counter,
              const NeatPolicy<Props>& policy, std::mt19937& rng);

template <typename Props>
void disable_connection(GraphGenome<Props>& genome, std::mt19937& rng);

// --- Composite ---

template <typename Props>
void mutate(GraphGenome<Props>& genome, InnovationCounter& counter,
            const NeatMutationConfig& config, const NeatPolicy<Props>& policy, std::mt19937& rng);

// --- Crossover & speciation ---

template <typename Props>
[[nodiscard]] GraphGenome<Props> crossover(
    const GraphGenome<Props>& fitter_parent, const GraphGenome<Props>& other_parent,
    const NeatPolicy<Props>& policy, std::mt19937& rng);

template <typename Props>
[[nodiscard]] float compatibility_distance(
    const GraphGenome<Props>& a, const GraphGenome<Props>& b, const SpeciationConfig& config);

// --- Genome factory ---

template <typename Props>
[[nodiscard]] GraphGenome<Props> create_minimal_genome(
    std::size_t num_inputs, std::size_t num_outputs,
    const NeatPolicy<Props>& policy, std::mt19937& rng);

} // namespace evolve

#include <evolve/neat_operators.inl>
