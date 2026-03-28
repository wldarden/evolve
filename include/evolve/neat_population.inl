#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>

namespace evolve {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

template <typename Props>
NeatPopulation<Props>::NeatPopulation(std::size_t num_inputs,
                                      std::size_t num_outputs,
                                      const NeatPopulationConfig& config,
                                      const NeatPolicy<Props>& policy,
                                      std::mt19937& rng)
    : config_(config), policy_(policy) {
    policy_.validate();

    // Seed the innovation counter with the initial connection innovations.
    // create_minimal_genome assigns innovations 0..num_inputs*num_outputs-1.
    for (std::size_t in = 0; in < num_inputs; ++in) {
        for (std::size_t out = 0; out < num_outputs; ++out) {
            innovation_counter_.get_or_create(
                static_cast<uint32_t>(in),
                static_cast<uint32_t>(num_inputs + out));
        }
    }
    innovation_counter_.new_generation();

    // Create the initial population of minimal genomes.
    individuals_.reserve(config_.population_size);
    for (std::size_t i = 0; i < config_.population_size; ++i) {
        NeatIndividual<Props> ind;
        ind.genome = create_minimal_genome<Props>(num_inputs, num_outputs,
                                                  policy_, rng);
        individuals_.push_back(std::move(ind));
    }

    // All individuals start in a single species.
    uint32_t sid = next_species_id_++;
    species_.push_back(SpeciesInfo{
        .id = sid,
        .representative = individuals_.front().genome,
    });
    for (auto& ind : individuals_) {
        ind.species_id = sid;
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template <typename Props>
std::vector<NeatIndividual<Props>>& NeatPopulation<Props>::individuals() {
    return individuals_;
}

template <typename Props>
const std::vector<NeatIndividual<Props>>&
NeatPopulation<Props>::individuals() const {
    return individuals_;
}

template <typename Props>
std::size_t NeatPopulation<Props>::generation() const noexcept {
    return generation_;
}

template <typename Props>
std::size_t NeatPopulation<Props>::num_species() const noexcept {
    return species_.size();
}

template <typename Props>
InnovationCounter& NeatPopulation<Props>::innovation_counter() noexcept {
    return innovation_counter_;
}

// ---------------------------------------------------------------------------
// Speciation
// ---------------------------------------------------------------------------

template <typename Props>
void NeatPopulation<Props>::speciate() {
    // Clear species membership; we'll reassign everyone.
    for (auto& ind : individuals_) {
        ind.species_id = 0;
    }

    // Track which species have at least one member this generation.
    std::vector<bool> species_has_members(species_.size(), false);

    for (auto& ind : individuals_) {
        bool placed = false;
        for (std::size_t s = 0; s < species_.size(); ++s) {
            float dist = compatibility_distance(
                ind.genome, species_[s].representative, config_.speciation);
            if (dist < config_.speciation.compatibility_threshold) {
                ind.species_id = species_[s].id;
                species_has_members[s] = true;
                placed = true;
                break;
            }
        }
        if (!placed) {
            // Create a new species for this individual.
            uint32_t sid = next_species_id_++;
            species_.push_back(SpeciesInfo{
                .id = sid,
                .representative = ind.genome,
            });
            species_has_members.push_back(true);
            ind.species_id = sid;
        }
    }

    // Remove species with no members.
    std::size_t write = 0;
    for (std::size_t read = 0; read < species_.size(); ++read) {
        if (species_has_members[read]) {
            if (write != read) {
                species_[write] = std::move(species_[read]);
            }
            ++write;
        }
    }
    species_.resize(write);

    // Update representatives: pick the first member found for each species.
    for (auto& sp : species_) {
        for (const auto& ind : individuals_) {
            if (ind.species_id == sp.id) {
                sp.representative = ind.genome;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stagnation elimination
// ---------------------------------------------------------------------------

template <typename Props>
void NeatPopulation<Props>::eliminate_stagnant_species() {
    // Update best fitness and stagnation counts.
    for (auto& sp : species_) {
        float species_best = 0.0f;
        for (const auto& ind : individuals_) {
            if (ind.species_id == sp.id) {
                species_best = std::max(species_best, ind.fitness);
            }
        }
        if (species_best > sp.best_fitness) {
            sp.best_fitness = species_best;
            sp.stagnation_count = 0;
        } else {
            ++sp.stagnation_count;
        }
    }

    // Sort species by stagnation count descending so we remove most stagnant
    // first.
    std::sort(species_.begin(), species_.end(),
              [](const SpeciesInfo& a, const SpeciesInfo& b) {
                  return a.stagnation_count > b.stagnation_count;
              });

    // Remove stagnant species from front, but keep at least
    // min_species_to_keep.
    std::size_t remaining = species_.size();
    std::size_t i = 0;
    while (i < species_.size() && remaining > config_.min_species_to_keep) {
        if (species_[i].stagnation_count >= config_.stagnation_limit) {
            // Remove this species: reassign its members nowhere (they'll be
            // removed).
            uint32_t dead_id = species_[i].id;
            species_.erase(species_.begin() +
                           static_cast<long>(i));
            // Remove individuals belonging to this species.
            individuals_.erase(
                std::remove_if(
                    individuals_.begin(), individuals_.end(),
                    [dead_id](const NeatIndividual<Props>& ind) {
                        return ind.species_id == dead_id;
                    }),
                individuals_.end());
            --remaining;
            // Don't increment i; the next species shifted into this position.
        } else {
            ++i;
        }
    }
}

// ---------------------------------------------------------------------------
// Evolve
// ---------------------------------------------------------------------------

template <typename Props>
void NeatPopulation<Props>::evolve(std::mt19937& rng) {
    // 1. Speciate
    speciate();

    // 2. Stagnation check
    eliminate_stagnant_species();

    // If all species removed (shouldn't happen with min_species_to_keep, but
    // safety), keep what we have.
    if (species_.empty() || individuals_.empty()) {
        ++generation_;
        innovation_counter_.new_generation();
        return;
    }

    // 3. Compute adjusted fitness = fitness / species_size
    //    and total adjusted fitness per species.
    struct SpeciesStats {
        uint32_t id;
        float total_adjusted_fitness;
        std::size_t member_count;
    };
    std::vector<SpeciesStats> stats;
    stats.reserve(species_.size());

    for (const auto& sp : species_) {
        SpeciesStats ss{sp.id, 0.0f, 0};
        for (const auto& ind : individuals_) {
            if (ind.species_id == sp.id) {
                ++ss.member_count;
            }
        }
        if (ss.member_count > 0) {
            for (const auto& ind : individuals_) {
                if (ind.species_id == sp.id) {
                    ss.total_adjusted_fitness +=
                        ind.fitness / static_cast<float>(ss.member_count);
                }
            }
        }
        stats.push_back(ss);
    }

    // 4. Offspring allocation proportional to total adjusted fitness.
    float global_adjusted_sum = 0.0f;
    for (const auto& ss : stats) {
        global_adjusted_sum += ss.total_adjusted_fitness;
    }

    std::vector<std::size_t> offspring_counts(stats.size(), 0);
    if (global_adjusted_sum > 0.0f) {
        std::size_t total_allocated = 0;
        for (std::size_t i = 0; i < stats.size(); ++i) {
            auto raw = static_cast<std::size_t>(
                (stats[i].total_adjusted_fitness / global_adjusted_sum) *
                static_cast<float>(config_.population_size));
            offspring_counts[i] = std::max(raw, std::size_t{1});
            total_allocated += offspring_counts[i];
        }
        // Adjust to match population_size exactly.
        // Give extra to species with highest adjusted fitness, or trim from
        // lowest.
        while (total_allocated < config_.population_size) {
            // Find species with highest adjusted fitness to give extras.
            std::size_t best = 0;
            for (std::size_t i = 1; i < stats.size(); ++i) {
                if (stats[i].total_adjusted_fitness >
                    stats[best].total_adjusted_fitness) {
                    best = i;
                }
            }
            offspring_counts[best]++;
            total_allocated++;
        }
        while (total_allocated > config_.population_size) {
            // Find species with lowest adjusted fitness (and count > 1) to
            // trim.
            std::size_t worst = stats.size(); // sentinel
            for (std::size_t i = 0; i < stats.size(); ++i) {
                if (offspring_counts[i] > 1) {
                    if (worst == stats.size() ||
                        stats[i].total_adjusted_fitness <
                            stats[worst].total_adjusted_fitness) {
                        worst = i;
                    }
                }
            }
            if (worst == stats.size()) break; // can't trim further
            offspring_counts[worst]--;
            total_allocated--;
        }
    } else {
        // All fitnesses zero -- distribute evenly.
        std::size_t per_species = config_.population_size / stats.size();
        std::size_t remainder = config_.population_size % stats.size();
        for (std::size_t i = 0; i < stats.size(); ++i) {
            offspring_counts[i] = per_species + (i < remainder ? 1 : 0);
        }
    }

    // 5. Reproduction
    std::vector<NeatIndividual<Props>> next_gen;
    next_gen.reserve(config_.population_size);

    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    for (std::size_t s = 0; s < stats.size(); ++s) {
        uint32_t sid = stats[s].id;
        std::size_t n_offspring = offspring_counts[s];

        // Gather members of this species, sorted by fitness descending.
        std::vector<std::size_t> member_indices;
        for (std::size_t i = 0; i < individuals_.size(); ++i) {
            if (individuals_[i].species_id == sid) {
                member_indices.push_back(i);
            }
        }
        std::sort(member_indices.begin(), member_indices.end(),
                  [this](std::size_t a, std::size_t b) {
                      return individuals_[a].fitness >
                             individuals_[b].fitness;
                  });

        if (member_indices.empty()) continue;

        // Elitism: top elitism_per_species pass through unchanged.
        std::size_t n_elites =
            std::min(config_.elitism_per_species, member_indices.size());
        n_elites = std::min(n_elites, n_offspring);
        for (std::size_t e = 0; e < n_elites; ++e) {
            NeatIndividual<Props> elite;
            elite.genome = individuals_[member_indices[e]].genome;
            elite.species_id = sid;
            next_gen.push_back(std::move(elite));
        }

        // Fill remaining offspring via crossover + mutate.
        std::size_t remaining_offspring = n_offspring - n_elites;
        std::uniform_int_distribution<std::size_t> member_pick(
            0, member_indices.size() - 1);

        for (std::size_t o = 0; o < remaining_offspring; ++o) {
            NeatIndividual<Props> child;
            child.species_id = sid;

            if (member_indices.size() >= 2 && prob(rng) < 0.75f) {
                // Crossover
                std::size_t p1_idx = member_indices[member_pick(rng)];
                std::size_t p2_idx = member_indices[member_pick(rng)];

                // Interspecies mating
                if (prob(rng) < config_.interspecies_mate_rate &&
                    individuals_.size() > member_indices.size()) {
                    std::uniform_int_distribution<std::size_t> global_pick(
                        0, individuals_.size() - 1);
                    p2_idx = global_pick(rng);
                }

                // Fitter parent goes first.
                if (individuals_[p1_idx].fitness >=
                    individuals_[p2_idx].fitness) {
                    child.genome = crossover(
                        individuals_[p1_idx].genome,
                        individuals_[p2_idx].genome, policy_, rng);
                } else {
                    child.genome = crossover(
                        individuals_[p2_idx].genome,
                        individuals_[p1_idx].genome, policy_, rng);
                }
            } else {
                // Asexual: clone a random member.
                child.genome =
                    individuals_[member_indices[member_pick(rng)]].genome;
            }

            mutate(child.genome, innovation_counter_, config_.mutation,
                   policy_, rng);
            next_gen.push_back(std::move(child));
        }
    }

    // 6. Replace population
    individuals_ = std::move(next_gen);
    innovation_counter_.new_generation();
    ++generation_;
}

} // namespace evolve
