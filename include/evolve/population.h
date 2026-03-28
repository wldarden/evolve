#pragma once

#include <evolve/genome.h>

#include <cstddef>
#include <random>
#include <vector>

namespace evolve {

struct EvolutionConfig {
    std::size_t genome_size = 0;
    std::size_t population_size = 100;
    MutationConfig mutation = {};
    std::size_t elitism_count = 2;
    std::size_t tournament_size = 5;
    float init_weight_min = -1.0f;
    float init_weight_max = 1.0f;
};

class Population {
public:
    /// Initialize with random genomes.
    Population(const EvolutionConfig& config, std::mt19937& rng);

    /// Run one generation: select parents, crossover, mutate, replace population.
    void evolve(std::mt19937& rng);

    /// Tournament selection: pick tournament_size random genomes, return the fittest.
    [[nodiscard]] const Genome& tournament_select(std::mt19937& rng) const;

    [[nodiscard]] std::size_t size() const noexcept { return genomes_.size(); }
    [[nodiscard]] Genome& genome(std::size_t index) { return genomes_[index]; }
    [[nodiscard]] const Genome& genome(std::size_t index) const { return genomes_[index]; }

private:
    EvolutionConfig config_;
    std::vector<Genome> genomes_;
};

} // namespace evolve
