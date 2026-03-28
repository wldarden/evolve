#include <evolve/neat_population.h>
#include <neuralnet/neural_node_props.h>
#include <neuralnet/neural_neat_policy.h>
#include <neuralnet/graph_network.h>

#include <gtest/gtest.h>
#include <random>

namespace ev = evolve;
namespace nn = neuralnet;

namespace {

nn::NeuralMutationConfig default_neural_config() { return {}; }
ev::NeatPolicy<nn::NeuralNodeProps> make_policy() {
    return nn::make_neural_neat_policy(default_neural_config(),
        nn::NodeType::Stateless, nn::Activation::Tanh);
}

} // namespace

TEST(NeatPopulationTest, ConstructCreatesPopulation) {
    ev::NeatPopulationConfig config;
    config.population_size = 50;
    std::mt19937 rng(42);
    ev::NeatPopulation<nn::NeuralNodeProps> pop(3, 2, config, make_policy(), rng);
    EXPECT_EQ(pop.individuals().size(), 50);
    EXPECT_EQ(pop.generation(), 0);
    EXPECT_GE(pop.num_species(), 1);
}

TEST(NeatPopulationTest, IndividualsAreValidNetworks) {
    ev::NeatPopulationConfig config;
    config.population_size = 20;
    std::mt19937 rng(42);
    ev::NeatPopulation<nn::NeuralNodeProps> pop(3, 2, config, make_policy(), rng);
    for (const auto& ind : pop.individuals()) {
        EXPECT_NO_THROW(nn::GraphNetwork{ind.genome});
        nn::GraphNetwork net(ind.genome);
        EXPECT_EQ(net.input_size(), 3);
        EXPECT_EQ(net.output_size(), 2);
    }
}

TEST(NeatPopulationTest, EvolveProducesNextGeneration) {
    ev::NeatPopulationConfig config;
    config.population_size = 30;
    std::mt19937 rng(42);
    ev::NeatPopulation<nn::NeuralNodeProps> pop(2, 1, config, make_policy(), rng);
    for (auto& ind : pop.individuals()) {
        ind.fitness = std::uniform_real_distribution<float>(0.0f, 10.0f)(rng);
    }
    pop.evolve(rng);
    EXPECT_EQ(pop.generation(), 1);
    EXPECT_EQ(pop.individuals().size(), 30);
    for (const auto& ind : pop.individuals()) {
        EXPECT_NO_THROW(nn::GraphNetwork{ind.genome});
    }
}

TEST(NeatPopulationTest, EvolveMultipleGenerations) {
    ev::NeatPopulationConfig config;
    config.population_size = 50;
    std::mt19937 rng(42);
    ev::NeatPopulation<nn::NeuralNodeProps> pop(3, 2, config, make_policy(), rng);
    for (int gen = 0; gen < 20; ++gen) {
        for (auto& ind : pop.individuals()) {
            ind.fitness = static_cast<float>(ind.genome.connections.size());
        }
        pop.evolve(rng);
    }
    EXPECT_EQ(pop.generation(), 20);
    EXPECT_EQ(pop.individuals().size(), 50);
    float avg_connections = 0;
    for (const auto& ind : pop.individuals()) {
        avg_connections += static_cast<float>(ind.genome.connections.size());
    }
    avg_connections /= static_cast<float>(pop.individuals().size());
    EXPECT_GT(avg_connections, 6.0f);
}

TEST(NeatPopulationTest, SpeciationCreatesMultipleSpecies) {
    ev::NeatPopulationConfig config;
    config.population_size = 100;
    config.speciation.compatibility_threshold = 1.0f;
    std::mt19937 rng(42);
    ev::NeatPopulation<nn::NeuralNodeProps> pop(3, 2, config, make_policy(), rng);
    for (int gen = 0; gen < 10; ++gen) {
        for (auto& ind : pop.individuals()) {
            ind.fitness = std::uniform_real_distribution<float>(0.0f, 10.0f)(rng);
        }
        pop.evolve(rng);
    }
    EXPECT_GT(pop.num_species(), 1);
}
