# evolve — Evolution Library

- **GitHub:** [wldarden/evolve](https://github.com/wldarden/evolve)
- **Dependencies:** None (standalone)
- **Used by:** neuralnet, NeuroFlyer, AntSim, EcoSim

Shared library providing genetic algorithms and NEAT neuroevolution. The NEAT subsystem is generic — templated over user-defined node properties.

## Two Evolution Systems

### Basic GA (`Genome` + `Population`)
Flat float-vector genomes with tournament selection, crossover, and Gaussian mutation. Used by NeuroFlyer for weight optimization within a fixed MLP topology.

```cpp
evolve::EvolutionConfig config;
config.population_size = 100;
config.genome_size = network_weight_count;
evolve::Population pop(config, rng);
// ... evaluate fitness ...
pop.evolve(rng);
```

### NEAT (`NeatPopulation<Props>`)
Full NEAT algorithm: topology-evolving graph networks with speciation. Templated over user-defined node properties (`Props`). Domain-specific node initialization, crossover, and mutation are provided by the consumer via a `NeatPolicy<Props>` callback struct.

```cpp
// Build a policy that knows how to initialize and mutate NeuralNodeProps
auto policy = neuralnet::make_neural_neat_policy(neuralnet::NeuralMutationConfig{});

evolve::NeatPopulationConfig config;
config.population_size = 150;

evolve::NeatPopulation<neuralnet::NeuralNodeProps> pop(
    num_inputs, num_outputs, config, policy, rng);

for (auto& ind : pop.individuals()) {
    neuralnet::GraphNetwork net(ind.genome);
    ind.fitness = evaluate(net);
}
pop.evolve(rng);
```

## NEAT Components

| Header | Contents |
|--------|----------|
| `evolve/neat_policy.h` | `NeatPolicy<Props>` — callback struct for init/merge/mutate node props |
| `evolve/innovation.h` | `InnovationCounter` — tracks structural mutation novelty |
| `evolve/neat_operators.h` | `NeatMutationConfig`, `SpeciationConfig`, all mutation/crossover/speciation functions |
| `evolve/neat_population.h` | `NeatIndividual<Props>`, `NeatPopulationConfig`, `NeatPopulation<Props>` class |

## NEAT Mutation Operators

**Parameter:** `mutate_weights` — Gaussian noise or full replacement on connection weights

**Structural:** `add_connection`, `add_node`, `disable_connection`

**Composite:** `mutate()` applies structural mutations according to config rates, then calls `policy.mutate_properties` for domain-specific mutations

Domain-specific mutations (bias, tau, node type, activation) are handled via `NeatPolicy<Props>` callbacks provided by the consumer (e.g. `neuralnet::make_neural_neat_policy()`).

## NEAT Population Lifecycle

Each call to `evolve()`:
1. **Speciate** — group by genetic similarity (compatibility distance)
2. **Stagnation** — eliminate species that haven't improved
3. **Adjusted fitness** — fitness / species_size (protects innovation)
4. **Offspring allocation** — proportional to species adjusted fitness
5. **Reproduce** — elitism + crossover + mutation within species

## Threading Model

`evolve()` is single-threaded. Evaluate fitness in parallel (your responsibility), then call `evolve()` sequentially.

## Basic GA Headers

| Header | Contents |
|--------|----------|
| `evolve/genome.h` | `Genome`, `MutationConfig`, `mutate()`, `crossover_uniform()` |
| `evolve/population.h` | `Population`, `EvolutionConfig` |
