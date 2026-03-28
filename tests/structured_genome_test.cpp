#include <evolve/structured_genome.h>

#include <gtest/gtest.h>

#include <cmath>

namespace ev = evolve;

// ============================================================
// Construction & Access
// ============================================================

TEST(StructuredGenome, AddGeneAndRetrieve) {
    ev::StructuredGenome g;
    g.add_gene({"speed", {2.5f}, {}});
    g.add_gene({"range", {100.0f}, {}});

    EXPECT_EQ(g.gene_count(), 2u);
    EXPECT_FLOAT_EQ(g.get("speed"), 2.5f);
    EXPECT_FLOAT_EQ(g.get("range"), 100.0f);
}

TEST(StructuredGenome, DuplicateTagThrows) {
    ev::StructuredGenome g;
    g.add_gene({"x", {1.0f}, {}});
    EXPECT_THROW(g.add_gene({"x", {2.0f}, {}}), std::runtime_error);
}

TEST(StructuredGenome, HasGene) {
    ev::StructuredGenome g;
    g.add_gene({"x", {1.0f}, {}});
    EXPECT_TRUE(g.has_gene("x"));
    EXPECT_FALSE(g.has_gene("y"));
}

TEST(StructuredGenome, GetSetSingleValue) {
    ev::StructuredGenome g;
    g.add_gene({"x", {1.0f}, {}});
    EXPECT_FLOAT_EQ(g.get("x"), 1.0f);
    g.set("x", 5.0f);
    EXPECT_FLOAT_EQ(g.get("x"), 5.0f);
}

TEST(StructuredGenome, GetOnMultiValueThrows) {
    ev::StructuredGenome g;
    g.add_gene({"weights", {1.0f, 2.0f, 3.0f}, {}});
    EXPECT_THROW((void)g.get("weights"), std::runtime_error);
    EXPECT_THROW(g.set("weights", 0.0f), std::runtime_error);
}

TEST(StructuredGenome, GeneNotFoundThrows) {
    ev::StructuredGenome g;
    EXPECT_THROW((void)g.gene("nope"), std::runtime_error);
    EXPECT_THROW((void)g.get("nope"), std::runtime_error);
}

TEST(StructuredGenome, MultiValueGene) {
    ev::StructuredGenome g;
    g.add_gene({"weights", {0.1f, 0.2f, 0.3f, 0.4f}, {}});
    EXPECT_EQ(g.gene("weights").values.size(), 4u);
    EXPECT_FLOAT_EQ(g.gene("weights").values[2], 0.3f);
}

// ============================================================
// Linkage Groups
// ============================================================

TEST(StructuredGenome, AddLinkageGroup) {
    ev::StructuredGenome g;
    g.add_gene({"s0_range", {100.0f}, {}});
    g.add_gene({"s0_width", {0.1f}, {}});
    g.add_linkage_group({"sensor_0", {"s0_range", "s0_width"}});

    EXPECT_EQ(g.linkage_groups().size(), 1u);
    EXPECT_EQ(g.linkage_groups()[0].gene_tags.size(), 2u);
}

TEST(StructuredGenome, LinkageGroupUnknownTagThrows) {
    ev::StructuredGenome g;
    g.add_gene({"x", {1.0f}, {}});
    EXPECT_THROW(
        g.add_linkage_group({"grp", {"x", "nonexistent"}}),
        std::runtime_error);
}

// ============================================================
// Flatten
// ============================================================

TEST(StructuredGenome, FlattenAll) {
    ev::StructuredGenome g;
    g.add_gene({"a", {1.0f, 2.0f}, {}});
    g.add_gene({"b", {3.0f}, {}});
    g.add_gene({"c", {4.0f, 5.0f, 6.0f}, {}});

    auto flat = g.flatten_all();
    ASSERT_EQ(flat.size(), 6u);
    EXPECT_FLOAT_EQ(flat[0], 1.0f);
    EXPECT_FLOAT_EQ(flat[1], 2.0f);
    EXPECT_FLOAT_EQ(flat[2], 3.0f);
    EXPECT_FLOAT_EQ(flat[3], 4.0f);
    EXPECT_FLOAT_EQ(flat[5], 6.0f);
}

TEST(StructuredGenome, FlattenByPrefix) {
    ev::StructuredGenome g;
    g.add_gene({"weight_L0", {1.0f, 2.0f}, {}});
    g.add_gene({"sensor_range", {100.0f}, {}});
    g.add_gene({"weight_L1", {3.0f, 4.0f, 5.0f}, {}});
    g.add_gene({"sensor_width", {0.1f}, {}});

    auto weights = g.flatten("weight_");
    ASSERT_EQ(weights.size(), 5u);
    EXPECT_FLOAT_EQ(weights[0], 1.0f);
    EXPECT_FLOAT_EQ(weights[2], 3.0f);
    EXPECT_FLOAT_EQ(weights[4], 5.0f);

    auto sensors = g.flatten("sensor_");
    ASSERT_EQ(sensors.size(), 2u);
    EXPECT_FLOAT_EQ(sensors[0], 100.0f);
    EXPECT_FLOAT_EQ(sensors[1], 0.1f);
}

TEST(StructuredGenome, FlattenNoMatch) {
    ev::StructuredGenome g;
    g.add_gene({"x", {1.0f}, {}});
    auto empty = g.flatten("zzz_");
    EXPECT_TRUE(empty.empty());
}

TEST(StructuredGenome, TotalValues) {
    ev::StructuredGenome g;
    g.add_gene({"a", {1.0f, 2.0f}, {}});
    g.add_gene({"b", {3.0f}, {}});
    EXPECT_EQ(g.total_values(), 3u);
}

// ============================================================
// Mutation
// ============================================================

TEST(StructuredGenome, MutationChangesValues) {
    ev::StructuredGenome g;
    g.add_gene({"x", {0.0f}, {.rate = 1.0f, .strength = 1.0f}});
    g.add_gene({"y", {0.0f}, {.rate = 1.0f, .strength = 1.0f}});

    auto original = g.flatten_all();

    std::mt19937 rng(42);
    ev::mutate(g, rng);

    auto mutated = g.flatten_all();
    EXPECT_NE(original, mutated);
}

TEST(StructuredGenome, MutationRespectsEvolvableFlag) {
    ev::StructuredGenome g;
    g.add_gene({"frozen", {5.0f}, {.rate = 1.0f, .strength = 10.0f, .evolvable = false}});
    g.add_gene({"free", {5.0f}, {.rate = 1.0f, .strength = 10.0f, .evolvable = true}});

    std::mt19937 rng(42);
    for (int i = 0; i < 100; ++i) {
        ev::mutate(g, rng);
    }

    EXPECT_FLOAT_EQ(g.get("frozen"), 5.0f);
    EXPECT_NE(g.get("free"), 5.0f);
}

TEST(StructuredGenome, MutationRespectsClamp) {
    ev::StructuredGenome g;
    g.add_gene({"x", {0.5f}, {.rate = 1.0f, .strength = 100.0f,
                               .min_val = 0.0f, .max_val = 1.0f}});

    std::mt19937 rng(42);
    for (int i = 0; i < 100; ++i) {
        ev::mutate(g, rng);
    }

    float val = g.get("x");
    EXPECT_GE(val, 0.0f);
    EXPECT_LE(val, 1.0f);
}

TEST(StructuredGenome, MutationPerGeneDifferentStrength) {
    // Gene with high strength should change more than gene with low strength
    float total_change_high = 0.0f;
    float total_change_low = 0.0f;
    constexpr int TRIALS = 200;

    for (int trial = 0; trial < TRIALS; ++trial) {
        ev::StructuredGenome g;
        g.add_gene({"high", {0.0f}, {.rate = 1.0f, .strength = 5.0f}});
        g.add_gene({"low", {0.0f}, {.rate = 1.0f, .strength = 0.01f}});

        std::mt19937 rng(static_cast<unsigned>(trial));
        ev::mutate(g, rng);

        total_change_high += std::abs(g.get("high"));
        total_change_low += std::abs(g.get("low"));
    }

    // High-strength gene should have accumulated much more total change
    EXPECT_GT(total_change_high, total_change_low * 10.0f);
}

// ============================================================
// Crossover
// ============================================================

TEST(StructuredGenome, CrossoverProducesChild) {
    ev::StructuredGenome a;
    a.add_gene({"x", {1.0f}, {}});
    a.add_gene({"y", {2.0f}, {}});

    ev::StructuredGenome b;
    b.add_gene({"x", {10.0f}, {}});
    b.add_gene({"y", {20.0f}, {}});

    std::mt19937 rng(42);
    auto child = ev::crossover(a, b, rng);

    EXPECT_EQ(child.gene_count(), 2u);
    // Each gene should come from either parent
    float cx = child.get("x");
    float cy = child.get("y");
    EXPECT_TRUE(cx == 1.0f || cx == 10.0f);
    EXPECT_TRUE(cy == 2.0f || cy == 20.0f);
}

TEST(StructuredGenome, CrossoverRespectsLinkageGroup) {
    // Two genes in a linkage group must ALWAYS come from the same parent.
    // Run many trials to verify they never get split.
    ev::StructuredGenome a;
    a.add_gene({"s0_range", {100.0f}, {}});
    a.add_gene({"s0_width", {0.1f}, {}});
    a.add_gene({"unrelated", {999.0f}, {}});
    a.add_linkage_group({"sensor_0", {"s0_range", "s0_width"}});

    ev::StructuredGenome b;
    b.add_gene({"s0_range", {200.0f}, {}});
    b.add_gene({"s0_width", {0.2f}, {}});
    b.add_gene({"unrelated", {888.0f}, {}});
    b.add_linkage_group({"sensor_0", {"s0_range", "s0_width"}});

    for (int trial = 0; trial < 200; ++trial) {
        std::mt19937 rng(static_cast<unsigned>(trial));
        auto child = ev::crossover(a, b, rng);

        float range = child.get("s0_range");
        float width = child.get("s0_width");

        // Both must come from the same parent
        if (range == 100.0f) {
            EXPECT_FLOAT_EQ(width, 0.1f)
                << "Trial " << trial << ": range from A but width not from A";
        } else {
            EXPECT_FLOAT_EQ(range, 200.0f);
            EXPECT_FLOAT_EQ(width, 0.2f)
                << "Trial " << trial << ": range from B but width not from B";
        }

        // Unrelated gene is independent — can be from either parent
        float u = child.get("unrelated");
        EXPECT_TRUE(u == 999.0f || u == 888.0f);
    }
}

TEST(StructuredGenome, CrossoverNonEvolvableAlwaysFromA) {
    ev::StructuredGenome a;
    a.add_gene({"locked", {42.0f}, {.evolvable = false}});
    a.add_gene({"free", {1.0f}, {}});

    ev::StructuredGenome b;
    b.add_gene({"locked", {99.0f}, {.evolvable = false}});
    b.add_gene({"free", {2.0f}, {}});

    for (int trial = 0; trial < 50; ++trial) {
        std::mt19937 rng(static_cast<unsigned>(trial));
        auto child = ev::crossover(a, b, rng);

        // Locked gene always from parent A
        EXPECT_FLOAT_EQ(child.get("locked"), 42.0f);
    }
}

TEST(StructuredGenome, CrossoverPreservesLinkageGroups) {
    ev::StructuredGenome a;
    a.add_gene({"x", {1.0f}, {}});
    a.add_gene({"y", {2.0f}, {}});
    a.add_linkage_group({"pair", {"x", "y"}});

    ev::StructuredGenome b;
    b.add_gene({"x", {10.0f}, {}});
    b.add_gene({"y", {20.0f}, {}});
    b.add_linkage_group({"pair", {"x", "y"}});

    std::mt19937 rng(42);
    auto child = ev::crossover(a, b, rng);

    EXPECT_EQ(child.linkage_groups().size(), 1u);
    EXPECT_EQ(child.linkage_groups()[0].name, "pair");
    EXPECT_EQ(child.linkage_groups()[0].gene_tags.size(), 2u);
}

TEST(StructuredGenome, CrossoverMultiValueGene) {
    // A gene with multiple values should come entirely from one parent
    ev::StructuredGenome a;
    a.add_gene({"weights", {1.0f, 2.0f, 3.0f}, {}});

    ev::StructuredGenome b;
    b.add_gene({"weights", {10.0f, 20.0f, 30.0f}, {}});

    for (int trial = 0; trial < 50; ++trial) {
        std::mt19937 rng(static_cast<unsigned>(trial));
        auto child = ev::crossover(a, b, rng);

        auto& vals = child.gene("weights").values;
        // All values should come from the same parent
        if (vals[0] == 1.0f) {
            EXPECT_FLOAT_EQ(vals[1], 2.0f);
            EXPECT_FLOAT_EQ(vals[2], 3.0f);
        } else {
            EXPECT_FLOAT_EQ(vals[0], 10.0f);
            EXPECT_FLOAT_EQ(vals[1], 20.0f);
            EXPECT_FLOAT_EQ(vals[2], 30.0f);
        }
    }
}

TEST(StructuredGenome, CrossoverBothParentsRepresented) {
    // Over many trials, both parents should contribute genes
    ev::StructuredGenome a;
    a.add_gene({"x", {1.0f}, {}});

    ev::StructuredGenome b;
    b.add_gene({"x", {2.0f}, {}});

    bool saw_a = false;
    bool saw_b = false;
    for (int trial = 0; trial < 100; ++trial) {
        std::mt19937 rng(static_cast<unsigned>(trial));
        auto child = ev::crossover(a, b, rng);
        if (child.get("x") == 1.0f) saw_a = true;
        if (child.get("x") == 2.0f) saw_b = true;
    }
    EXPECT_TRUE(saw_a);
    EXPECT_TRUE(saw_b);
}

// ============================================================
// Integration: realistic NeuroFlyer-like genome
// ============================================================

TEST(StructuredGenome, RealisticNeuroFlyerGenome) {
    ev::StructuredGenome g;

    // 5 sensors with range + width, linked
    for (int i = 0; i < 5; ++i) {
        std::string prefix = "sensor_" + std::to_string(i);
        g.add_gene({prefix + "_range", {120.0f},
                    {.rate = 0.05f, .strength = 10.0f, .min_val = 30.0f, .max_val = 400.0f}});
        g.add_gene({prefix + "_width", {0.15f},
                    {.rate = 0.05f, .strength = 0.02f, .min_val = 0.02f, .max_val = 0.6f}});
        g.add_linkage_group({prefix, {prefix + "_range", prefix + "_width"}});
    }

    // Network weights (2 layers)
    g.add_gene({"weight_L0", std::vector<float>(100, 0.0f),
                {.rate = 0.1f, .strength = 0.3f}});
    g.add_gene({"weight_L1", std::vector<float>(50, 0.0f),
                {.rate = 0.1f, .strength = 0.3f}});

    // Memory slots (frozen for now)
    g.add_gene({"memory_slots", {4.0f},
                {.rate = 0.01f, .strength = 1.0f, .min_val = 0.0f, .max_val = 16.0f,
                 .evolvable = false}});

    EXPECT_EQ(g.gene_count(), 13u);  // 5*2 sensor + 2 weight + 1 memory
    EXPECT_EQ(g.total_values(), 161u);  // 10 sensor + 150 weight + 1 memory
    EXPECT_EQ(g.linkage_groups().size(), 5u);

    // Flatten weights only
    auto weights = g.flatten("weight_");
    EXPECT_EQ(weights.size(), 150u);

    // Flatten sensor params only
    auto sensors = g.flatten("sensor_");
    EXPECT_EQ(sensors.size(), 10u);

    // Mutate
    std::mt19937 rng(42);
    auto before = g.flatten_all();
    ev::mutate(g, rng);
    auto after = g.flatten_all();
    EXPECT_NE(before, after);

    // Memory should be unchanged (frozen)
    EXPECT_FLOAT_EQ(g.get("memory_slots"), 4.0f);

    // Sensor values should still be in range
    for (int i = 0; i < 5; ++i) {
        float range = g.get("sensor_" + std::to_string(i) + "_range");
        float width = g.get("sensor_" + std::to_string(i) + "_width");
        EXPECT_GE(range, 30.0f);
        EXPECT_LE(range, 400.0f);
        EXPECT_GE(width, 0.02f);
        EXPECT_LE(width, 0.6f);
    }
}
