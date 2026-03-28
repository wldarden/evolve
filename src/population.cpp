#include <evolve/population.h>

#include <algorithm>
#include <cassert>

namespace evolve {

Population::Population(const EvolutionConfig& config, std::mt19937& rng) : config_(config) {
    genomes_.reserve(config.population_size);
    for (std::size_t i = 0; i < config.population_size; ++i) {
        genomes_.push_back(
            Genome::random(config.genome_size, config.init_weight_min, config.init_weight_max, rng));
    }
}

const Genome& Population::tournament_select(std::mt19937& rng) const {
    assert(!genomes_.empty());

    // When the tournament covers the whole population, return the global best
    // deterministically (avoids sampling bias with replacement).
    if (config_.tournament_size >= genomes_.size()) {
        auto it = std::max_element(genomes_.begin(), genomes_.end(),
                                   [](const Genome& a, const Genome& b) {
                                       return a.fitness() < b.fitness();
                                   });
        return *it;
    }

    std::uniform_int_distribution<std::size_t> dist(0, genomes_.size() - 1);
    std::size_t best_idx = dist(rng);
    for (std::size_t i = 1; i < config_.tournament_size; ++i) {
        auto candidate = dist(rng);
        if (genomes_[candidate].fitness() > genomes_[best_idx].fitness()) {
            best_idx = candidate;
        }
    }
    return genomes_[best_idx];
}

void Population::evolve(std::mt19937& rng) {
    // Sort by fitness descending for elitism
    std::sort(genomes_.begin(), genomes_.end(),
              [](const Genome& a, const Genome& b) { return a.fitness() > b.fitness(); });

    std::vector<Genome> next_gen;
    next_gen.reserve(config_.population_size);

    // Elitism: carry over top N unchanged
    for (std::size_t i = 0; i < config_.elitism_count && i < genomes_.size(); ++i) {
        next_gen.push_back(Genome(genomes_[i].genes()));
    }

    // Fill the rest with offspring
    while (next_gen.size() < config_.population_size) {
        const auto& parent_a = tournament_select(rng);
        const auto& parent_b = tournament_select(rng);
        auto child = crossover_uniform(parent_a, parent_b, rng);
        mutate(child, config_.mutation, rng);
        next_gen.push_back(std::move(child));
    }

    genomes_ = std::move(next_gen);
}

} // namespace evolve
