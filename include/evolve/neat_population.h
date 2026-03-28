#pragma once
#include <evolve/innovation.h>
#include <evolve/neat_operators.h>

#include <cstdint>
#include <random>
#include <vector>

namespace evolve {

template <typename Props>
struct NeatIndividual {
    GraphGenome<Props> genome;
    float fitness = 0.0f;
    uint32_t species_id = 0;
};

struct NeatPopulationConfig {
    std::size_t population_size = 150;
    NeatMutationConfig mutation;
    SpeciationConfig speciation;
    std::size_t elitism_per_species = 1;
    float interspecies_mate_rate = 0.001f;
    std::size_t stagnation_limit = 15;
    std::size_t min_species_to_keep = 2;
};

template <typename Props>
class NeatPopulation {
public:
    NeatPopulation(std::size_t num_inputs, std::size_t num_outputs,
                   const NeatPopulationConfig& config,
                   const NeatPolicy<Props>& policy,
                   std::mt19937& rng);

    [[nodiscard]] std::vector<NeatIndividual<Props>>& individuals();
    [[nodiscard]] const std::vector<NeatIndividual<Props>>& individuals() const;
    void evolve(std::mt19937& rng);
    [[nodiscard]] std::size_t generation() const noexcept;
    [[nodiscard]] std::size_t num_species() const noexcept;
    [[nodiscard]] InnovationCounter& innovation_counter() noexcept;

private:
    struct SpeciesInfo {
        uint32_t id;
        GraphGenome<Props> representative;
        float best_fitness = 0.0f;
        std::size_t stagnation_count = 0;
    };

    void speciate();
    void eliminate_stagnant_species();

    NeatPopulationConfig config_;
    NeatPolicy<Props> policy_;
    std::vector<NeatIndividual<Props>> individuals_;
    std::vector<SpeciesInfo> species_;
    InnovationCounter innovation_counter_;
    std::size_t generation_ = 0;
    uint32_t next_species_id_ = 0;
};

} // namespace evolve

#include <evolve/neat_population.inl>
