#include <evolve/genome.h>

#include <cassert>

namespace evolve {

Genome::Genome(std::size_t size) : genes_(size, 0.0f) {}

Genome::Genome(std::vector<float> genes) : genes_(std::move(genes)) {}

Genome Genome::random(std::size_t size, float min, float max, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(min, max);
    std::vector<float> genes(size);
    for (auto& g : genes) {
        g = dist(rng);
    }
    return Genome(std::move(genes));
}

void mutate(Genome& genome, const MutationConfig& config, std::mt19937& rng) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, config.strength);

    for (auto& gene : genome.genes()) {
        if (chance(rng) < config.rate) {
            gene += noise(rng);
        }
    }
}

Genome crossover_uniform(const Genome& a, const Genome& b, std::mt19937& rng) {
    assert(a.genes().size() == b.genes().size());

    std::uniform_int_distribution<int> coin(0, 1);
    std::vector<float> child_genes(a.genes().size());

    for (std::size_t i = 0; i < child_genes.size(); ++i) {
        child_genes[i] = coin(rng) ? a.genes()[i] : b.genes()[i];
    }

    return Genome(std::move(child_genes));
}

Genome crossover_single_point(const Genome& a, const Genome& b, std::mt19937& rng) {
    assert(a.genes().size() == b.genes().size());
    const auto size = a.genes().size();

    std::uniform_int_distribution<std::size_t> dist(1, size - 1);
    const auto split = dist(rng);

    std::vector<float> child_genes(size);
    for (std::size_t i = 0; i < split; ++i) {
        child_genes[i] = a.genes()[i];
    }
    for (std::size_t i = split; i < size; ++i) {
        child_genes[i] = b.genes()[i];
    }

    return Genome(std::move(child_genes));
}

} // namespace evolve
