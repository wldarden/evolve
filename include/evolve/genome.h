#pragma once

#include <random>
#include <span>
#include <vector>

namespace evolve {

struct MutationConfig {
    float rate = 0.1f;      // probability of each gene being mutated
    float strength = 0.3f;  // std deviation of Gaussian noise
};

class Genome {
public:
    /// Construct with a given size, all genes zeroed.
    explicit Genome(std::size_t size);

    /// Construct from existing gene values.
    explicit Genome(std::vector<float> genes);

    /// Create a genome with random values uniformly distributed in [min, max].
    [[nodiscard]] static Genome random(std::size_t size, float min, float max,
                                       std::mt19937& rng);

    [[nodiscard]] std::vector<float>& genes() noexcept { return genes_; }
    [[nodiscard]] const std::vector<float>& genes() const noexcept { return genes_; }

    [[nodiscard]] float fitness() const noexcept { return fitness_; }
    void set_fitness(float f) noexcept { fitness_ = f; }

private:
    std::vector<float> genes_;
    float fitness_ = 0.0f;
};

/// Apply Gaussian mutation in-place.
void mutate(Genome& genome, const MutationConfig& config, std::mt19937& rng);

/// Uniform crossover: each gene randomly from parent a or b.
[[nodiscard]] Genome crossover_uniform(const Genome& a, const Genome& b, std::mt19937& rng);

/// Single-point crossover: prefix from a, suffix from b (split point random).
[[nodiscard]] Genome crossover_single_point(const Genome& a, const Genome& b, std::mt19937& rng);

} // namespace evolve
