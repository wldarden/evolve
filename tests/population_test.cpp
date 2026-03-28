#include <evolve/population.h>

#include <gtest/gtest.h>

#include <algorithm>

namespace ev = evolve;

TEST(PopulationTest, InitializeWithRandomGenomes) {
    ev::EvolutionConfig config{
        .genome_size = 10,
        .population_size = 50,
    };

    std::mt19937 rng(42);
    ev::Population pop(config, rng);

    EXPECT_EQ(pop.size(), 50);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        EXPECT_EQ(pop.genome(i).genes().size(), 10);
    }
}

TEST(PopulationTest, TournamentSelectionPicksFittest) {
    ev::EvolutionConfig config{
        .genome_size = 5,
        .population_size = 10,
        .tournament_size = 10,  // entire population = always picks best
    };

    std::mt19937 rng(42);
    ev::Population pop(config, rng);

    // Set fitness: genome 7 is the best
    for (std::size_t i = 0; i < pop.size(); ++i) {
        pop.genome(i).set_fitness(static_cast<float>(i));
    }

    auto& selected = pop.tournament_select(rng);
    EXPECT_FLOAT_EQ(selected.fitness(), 9.0f);
}

TEST(PopulationTest, EvolvePreservesElites) {
    ev::EvolutionConfig config{
        .genome_size = 5,
        .population_size = 20,
        .mutation = {.rate = 0.5f, .strength = 1.0f},
        .elitism_count = 2,
        .tournament_size = 3,
    };

    std::mt19937 rng(42);
    ev::Population pop(config, rng);

    // Assign fitness
    for (std::size_t i = 0; i < pop.size(); ++i) {
        pop.genome(i).set_fitness(static_cast<float>(i));
    }

    // Remember the top 2 genomes
    auto best_genes = pop.genome(19).genes();
    auto second_genes = pop.genome(18).genes();

    pop.evolve(rng);

    // After evolution, the top 2 should still be present (elitism)
    EXPECT_EQ(pop.size(), 20);

    bool found_best = false;
    bool found_second = false;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        if (pop.genome(i).genes() == best_genes) found_best = true;
        if (pop.genome(i).genes() == second_genes) found_second = true;
    }
    EXPECT_TRUE(found_best);
    EXPECT_TRUE(found_second);
}

TEST(PopulationTest, EvolveProducesSamePopulationSize) {
    ev::EvolutionConfig config{
        .genome_size = 10,
        .population_size = 30,
        .tournament_size = 3,
    };

    std::mt19937 rng(42);
    ev::Population pop(config, rng);

    for (std::size_t i = 0; i < pop.size(); ++i) {
        pop.genome(i).set_fitness(static_cast<float>(i));
    }

    pop.evolve(rng);
    EXPECT_EQ(pop.size(), 30);
}
