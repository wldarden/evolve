#include <evolve/neat_operators.h>
#include <evolve/innovation.h>
#include <evolve/node_role.h>
#include <evolve/graph_gene.h>
#include <neuralnet/neural_node_props.h>
#include <neuralnet/neural_neat_policy.h>
#include <neuralnet/graph_network.h>

#include <gtest/gtest.h>
#include <random>

namespace ev = evolve;
namespace nn = neuralnet;

namespace {

nn::NeuralMutationConfig default_neural_config() { return {}; }
ev::NeatPolicy<nn::NeuralNodeProps> default_neural_policy() {
    return nn::make_neural_neat_policy(default_neural_config());
}

nn::NeuralGenome make_test_genome() {
    nn::NeuralGenome g;
    g.nodes = {
        {.id = 0, .role = ev::NodeRole::Input,  .props = {.activation = nn::Activation::ReLU,  .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 1, .role = ev::NodeRole::Input,  .props = {.activation = nn::Activation::ReLU,  .type = nn::NodeType::Stateless, .bias = 0.0f, .tau = 1.0f}},
        {.id = 2, .role = ev::NodeRole::Hidden, .props = {.activation = nn::Activation::Tanh,  .type = nn::NodeType::CTRNN,     .bias = 0.5f, .tau = 5.0f}},
        {.id = 3, .role = ev::NodeRole::Output, .props = {.activation = nn::Activation::Tanh,  .type = nn::NodeType::Stateless, .bias = 0.1f, .tau = 1.0f}},
    };
    g.connections = {
        {.from_node = 0, .to_node = 2, .weight = 1.0f,  .enabled = true, .innovation = 0},
        {.from_node = 1, .to_node = 2, .weight = -0.5f, .enabled = true, .innovation = 1},
        {.from_node = 2, .to_node = 3, .weight = 0.8f,  .enabled = true, .innovation = 2},
    };
    return g;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test 1: MutateWeights — all connections perturbed with rate=1.0
// ---------------------------------------------------------------------------
TEST(NeatMutation, MutateWeights_AllPerturbed) {
    auto genome = make_test_genome();
    auto original = genome;

    ev::NeatWeightConfig config;
    config.weight_mutate_rate = 1.0f;   // every connection
    config.weight_perturb_rate = 1.0f;  // always perturb (never replace)

    std::mt19937 rng(42);
    ev::mutate_weights(genome, config, rng);

    bool any_changed = false;
    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        if (genome.connections[i].weight != original.connections[i].weight) {
            any_changed = true;
        }
    }
    EXPECT_TRUE(any_changed);
}

// ---------------------------------------------------------------------------
// Test 2: MutateWeights — zero rate changes nothing
// ---------------------------------------------------------------------------
TEST(NeatMutation, MutateWeights_ZeroRateChangesNothing) {
    auto genome = make_test_genome();
    auto original = genome;

    ev::NeatWeightConfig config;
    config.weight_mutate_rate = 0.0f;

    std::mt19937 rng(42);
    ev::mutate_weights(genome, config, rng);

    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        EXPECT_FLOAT_EQ(genome.connections[i].weight, original.connections[i].weight);
    }
}

// ---------------------------------------------------------------------------
// Test 3: MutateBiases — changes non-input nodes
// ---------------------------------------------------------------------------
TEST(NeatMutation, MutateBiases_ChangesNonInputNodes) {
    auto genome = make_test_genome();
    auto original = genome;

    nn::NeuralMutationConfig neural_config;
    neural_config.bias_mutate_rate = 1.0f;  // every eligible node

    std::mt19937 rng(42);
    nn::mutate_biases(genome, neural_config, rng);

    // Input nodes should be unchanged.
    EXPECT_FLOAT_EQ(genome.nodes[0].props.bias, original.nodes[0].props.bias);
    EXPECT_FLOAT_EQ(genome.nodes[1].props.bias, original.nodes[1].props.bias);

    // At least one non-input node should have changed.
    bool any_changed = false;
    for (std::size_t i = 2; i < genome.nodes.size(); ++i) {
        if (genome.nodes[i].props.bias != original.nodes[i].props.bias) {
            any_changed = true;
        }
    }
    EXPECT_TRUE(any_changed);
}

// ---------------------------------------------------------------------------
// Test 4: MutateTau — only affects CTRNN nodes
// ---------------------------------------------------------------------------
TEST(NeatMutation, MutateTau_OnlyCTRNNNodes) {
    auto genome = make_test_genome();
    auto original = genome;

    nn::NeuralMutationConfig neural_config;
    neural_config.tau_mutate_rate = 1.0f;

    std::mt19937 rng(42);
    nn::mutate_tau(genome, neural_config, rng);

    // Node 2 is CTRNN — its tau should have changed.
    EXPECT_NE(genome.nodes[2].props.tau, original.nodes[2].props.tau);

    // Non-CTRNN nodes should be unchanged.
    EXPECT_FLOAT_EQ(genome.nodes[0].props.tau, original.nodes[0].props.tau);
    EXPECT_FLOAT_EQ(genome.nodes[1].props.tau, original.nodes[1].props.tau);
    EXPECT_FLOAT_EQ(genome.nodes[3].props.tau, original.nodes[3].props.tau);

    // Tau should be clamped within range.
    EXPECT_GE(genome.nodes[2].props.tau, neural_config.tau_min);
    EXPECT_LE(genome.nodes[2].props.tau, neural_config.tau_max);
}

// ---------------------------------------------------------------------------
// Test 5: AddConnection — increases connection count
// ---------------------------------------------------------------------------
TEST(NeatMutation, AddConnection_IncreasesConnectionCount) {
    auto genome = make_test_genome();
    std::size_t before = genome.connections.size();

    ev::InnovationCounter counter;
    // Seed counter past existing innovations.
    counter.get_or_create(0, 2);  // innov 0
    counter.get_or_create(1, 2);  // innov 1
    counter.get_or_create(2, 3);  // innov 2

    std::mt19937 rng(42);

    ev::add_connection(genome, counter, rng);

    EXPECT_EQ(genome.connections.size(), before + 1);
    // New connection should be enabled.
    EXPECT_TRUE(genome.connections.back().enabled);
    // Target should not be an input node.
    auto to_id = genome.connections.back().to_node;
    for (const auto& n : genome.nodes) {
        if (n.id == to_id) {
            EXPECT_NE(n.role, ev::NodeRole::Input);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 6: AddNode — splits a connection
// ---------------------------------------------------------------------------
TEST(NeatMutation, AddNode_SplitsConnection) {
    auto genome = make_test_genome();
    std::size_t nodes_before = genome.nodes.size();
    std::size_t conns_before = genome.connections.size();

    ev::InnovationCounter counter;
    counter.get_or_create(0, 2);
    counter.get_or_create(1, 2);
    counter.get_or_create(2, 3);

    std::mt19937 rng(42);
    ev::add_node(genome, counter, default_neural_policy(), rng);

    // One new node, two new connections.
    EXPECT_EQ(genome.nodes.size(), nodes_before + 1);
    EXPECT_EQ(genome.connections.size(), conns_before + 2);

    // New node should be Hidden.
    EXPECT_EQ(genome.nodes.back().role, ev::NodeRole::Hidden);

    // Exactly one connection should have been disabled.
    int disabled_count = 0;
    for (const auto& c : genome.connections) {
        if (!c.enabled) ++disabled_count;
    }
    EXPECT_EQ(disabled_count, 1);
}

// ---------------------------------------------------------------------------
// Test 7: DisableConnection — disables exactly one
// ---------------------------------------------------------------------------
TEST(NeatMutation, DisableConnection_DisablesOne) {
    auto genome = make_test_genome();
    // All 3 connections start enabled.

    std::mt19937 rng(42);
    ev::disable_connection(genome, rng);

    int enabled_count = 0;
    for (const auto& c : genome.connections) {
        if (c.enabled) ++enabled_count;
    }
    EXPECT_EQ(enabled_count, 2);
}

// ---------------------------------------------------------------------------
// Test 8: MutateNodeType — toggles CTRNN<->Stateless
// ---------------------------------------------------------------------------
TEST(NeatMutation, MutateNodeType_TogglesCTRNN) {
    auto genome = make_test_genome();

    nn::NeuralMutationConfig neural_config;
    neural_config.node_type_mutate_rate = 1.0f;  // mutate every eligible node

    std::mt19937 rng(42);
    nn::mutate_node_types(genome, neural_config, rng);

    // Input nodes should not change.
    EXPECT_EQ(genome.nodes[0].props.type, nn::NodeType::Stateless);
    EXPECT_EQ(genome.nodes[1].props.type, nn::NodeType::Stateless);

    // Node 2 was CTRNN -> should now be Stateless.
    EXPECT_EQ(genome.nodes[2].props.type, nn::NodeType::Stateless);

    // Node 3 was Stateless -> should now be CTRNN with tau=1.0.
    EXPECT_EQ(genome.nodes[3].props.type, nn::NodeType::CTRNN);
    EXPECT_FLOAT_EQ(genome.nodes[3].props.tau, 1.0f);
}

// ---------------------------------------------------------------------------
// Test 9: MutateActivation — changes hidden and output
// ---------------------------------------------------------------------------
TEST(NeatMutation, MutateActivation_ChangesHiddenAndOutput) {
    auto genome = make_test_genome();

    nn::NeuralMutationConfig neural_config;
    neural_config.activation_mutate_rate = 1.0f;

    // Run multiple times to increase odds of an actual change (random pick
    // might land on the same activation).
    std::mt19937 rng(42);
    bool any_changed = false;
    for (int trial = 0; trial < 20; ++trial) {
        auto g = make_test_genome();
        nn::mutate_activations(g, neural_config, rng);

        // Inputs must never change.
        EXPECT_EQ(g.nodes[0].props.activation, nn::Activation::ReLU);
        EXPECT_EQ(g.nodes[1].props.activation, nn::Activation::ReLU);

        if (g.nodes[2].props.activation != nn::Activation::Tanh ||
            g.nodes[3].props.activation != nn::Activation::Tanh) {
            any_changed = true;
        }
    }
    EXPECT_TRUE(any_changed);
}

// ---------------------------------------------------------------------------
// Test 10: FullMutate — does not crash, genome still builds a network
// ---------------------------------------------------------------------------
TEST(NeatMutation, FullMutate_DoesNotCrash) {
    ev::NeatMutationConfig ev_config;
    ev_config.add_connection_rate = 0.30f;
    ev_config.add_node_rate       = 0.10f;

    ev::InnovationCounter counter;
    counter.get_or_create(0, 2);
    counter.get_or_create(1, 2);
    counter.get_or_create(2, 3);

    std::mt19937 rng(123);

    auto genome = make_test_genome();
    for (int i = 0; i < 100; ++i) {
        ev::mutate(genome, counter, ev_config, default_neural_policy(), rng);
    }

    // The mutated genome should still be constructible as a network.
    EXPECT_NO_THROW(nn::GraphNetwork{genome});
}

// === Crossover Tests ===

TEST(NeatCrossoverTest, MatchingGenes_InheritedFromEitherParent) {
    auto parent_a = make_test_genome();
    auto parent_b = make_test_genome();
    for (auto& conn : parent_b.connections) conn.weight += 10.0f;

    std::mt19937 rng(42);
    auto child = ev::crossover(parent_a, parent_b, default_neural_policy(), rng);
    EXPECT_EQ(child.connections.size(), 3);
    for (std::size_t i = 0; i < child.connections.size(); ++i) {
        float w = child.connections[i].weight;
        EXPECT_TRUE(w == parent_a.connections[i].weight || w == parent_b.connections[i].weight);
    }
}

TEST(NeatCrossoverTest, DisjointGenes_InheritedFromFitterParent) {
    auto fitter = make_test_genome();
    auto other = make_test_genome();
    fitter.connections.push_back(ev::ConnectionGene{
        .from_node = 0, .to_node = 3, .weight = 99.0f, .enabled = true, .innovation = 10});
    std::mt19937 rng(42);
    auto child = ev::crossover(fitter, other, default_neural_policy(), rng);
    EXPECT_EQ(child.connections.size(), 4);
    bool found = false;
    for (const auto& conn : child.connections) {
        if (conn.innovation == 10) { found = true; EXPECT_FLOAT_EQ(conn.weight, 99.0f); }
    }
    EXPECT_TRUE(found);
}

TEST(NeatCrossoverTest, DisabledGene_75PercentChanceStaysDisabled) {
    auto fitter = make_test_genome();
    auto other = make_test_genome();
    fitter.connections[0].enabled = false;
    int disabled_count = 0;
    for (int i = 0; i < 1000; ++i) {
        std::mt19937 rng(static_cast<unsigned>(i));
        auto child = ev::crossover(fitter, other, default_neural_policy(), rng);
        if (!child.connections[0].enabled) disabled_count++;
    }
    float ratio = static_cast<float>(disabled_count) / 1000.0f;
    EXPECT_NEAR(ratio, 0.75f, 0.10f);
}

TEST(NeatCrossoverTest, NodeProperties_InheritedRandomly) {
    auto fitter = make_test_genome();
    auto other = make_test_genome();
    fitter.nodes[2].props.bias = 1.0f;
    other.nodes[2].props.bias = -1.0f;
    bool got_fitter = false, got_other = false;
    for (int i = 0; i < 100; ++i) {
        std::mt19937 rng(static_cast<unsigned>(i));
        auto child = ev::crossover(fitter, other, default_neural_policy(), rng);
        for (const auto& node : child.nodes) {
            if (node.id == 2) {
                if (node.props.bias == 1.0f) got_fitter = true;
                if (node.props.bias == -1.0f) got_other = true;
            }
        }
    }
    EXPECT_TRUE(got_fitter);
    EXPECT_TRUE(got_other);
}

// === Speciation Tests ===

TEST(SpeciationTest, IdenticalGenomes_ZeroDistance) {
    auto a = make_test_genome();
    ev::SpeciationConfig config;
    EXPECT_FLOAT_EQ(ev::compatibility_distance(a, a, config), 0.0f);
}

TEST(SpeciationTest, DifferentWeights_NonZeroDistance) {
    auto a = make_test_genome();
    auto b = make_test_genome();
    b.connections[0].weight += 2.0f;
    ev::SpeciationConfig config;
    EXPECT_GT(ev::compatibility_distance(a, b, config), 0.0f);
}

TEST(SpeciationTest, ExtraConnections_IncreasesDistance) {
    auto a = make_test_genome();
    auto b = make_test_genome();
    b.connections.push_back({.from_node = 0, .to_node = 3, .weight = 1.0f, .enabled = true, .innovation = 10});
    b.connections.push_back({.from_node = 1, .to_node = 3, .weight = 1.0f, .enabled = true, .innovation = 11});
    ev::SpeciationConfig config;
    EXPECT_GT(ev::compatibility_distance(a, b, config), 0.0f);
}

TEST(SpeciationTest, SmallGenomes_NNotNormalized) {
    auto a = make_test_genome();
    auto b = make_test_genome();
    b.connections.push_back({.from_node = 0, .to_node = 3, .weight = 1.0f, .enabled = true, .innovation = 10});
    ev::SpeciationConfig config;
    config.c_excess = 1.0f;
    config.c_disjoint = 1.0f;
    config.c_weight = 0.0f;
    auto dist = ev::compatibility_distance(a, b, config);
    EXPECT_FLOAT_EQ(dist, 1.0f);
}
