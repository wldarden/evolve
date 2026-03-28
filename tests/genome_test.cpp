#include <evolve/genome.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace ev = evolve;

TEST(GenomeTest, ConstructWithSize) {
    ev::Genome g(10);
    EXPECT_EQ(g.genes().size(), 10);
    EXPECT_FLOAT_EQ(g.fitness(), 0.0f);
}

TEST(GenomeTest, ConstructWithValues) {
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    ev::Genome g(vals);
    EXPECT_EQ(g.genes(), vals);
}

TEST(GenomeTest, RandomInitProducesValuesInRange) {
    std::mt19937 rng(42);
    auto g = ev::Genome::random(100, -1.0f, 1.0f, rng);
    for (float v : g.genes()) {
        EXPECT_GE(v, -1.0f);
        EXPECT_LE(v, 1.0f);
    }
}

TEST(GenomeTest, MutateChangesAtLeastOneGene) {
    std::mt19937 rng(42);
    auto g = ev::Genome::random(100, -1.0f, 1.0f, rng);
    auto original = g.genes();

    ev::MutationConfig config{.rate = 1.0f, .strength = 0.5f};
    ev::mutate(g, config, rng);

    EXPECT_NE(g.genes(), original);
}

TEST(GenomeTest, MutateWithZeroRateChangesNothing) {
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    ev::Genome g(vals);

    ev::MutationConfig config{.rate = 0.0f, .strength = 0.5f};
    std::mt19937 rng(42);
    ev::mutate(g, config, rng);

    EXPECT_EQ(g.genes(), vals);
}

TEST(GenomeTest, CrossoverUniformProducesCorrectSize) {
    ev::Genome a(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    ev::Genome b(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f});

    std::mt19937 rng(42);
    auto child = ev::crossover_uniform(a, b, rng);

    EXPECT_EQ(child.genes().size(), 4);
    // Every gene should come from parent a or b
    for (std::size_t i = 0; i < 4; ++i) {
        float v = child.genes()[i];
        EXPECT_TRUE(v == a.genes()[i] || v == b.genes()[i]);
    }
}

TEST(GenomeTest, CrossoverSinglePoint) {
    ev::Genome a(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f});
    ev::Genome b(std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f});

    std::mt19937 rng(42);
    auto child = ev::crossover_single_point(a, b, rng);

    EXPECT_EQ(child.genes().size(), 4);
    // Should have a prefix from one parent and suffix from the other
    bool found_switch = false;
    for (std::size_t i = 1; i < 4; ++i) {
        if (child.genes()[i] != child.genes()[i - 1]) {
            found_switch = true;
        }
    }
    EXPECT_TRUE(found_switch);
}
