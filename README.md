# evolve — Evolutionary Algorithms Library

Shared library providing evolutionary algorithm primitives for neuroevolution projects. Supports two independent evolutionary systems with different complexity/flexibility trade-offs.

## Evolutionary Systems

### Basic Genetic Algorithm

A simple, fast GA operating on flat float-vector genomes. Good for fixed-topology networks (e.g. training MLP weights).

**Selection:** Tournament selection — picks `tournament_size` random individuals and returns the fittest.

**Crossover:**
- **Uniform** — each gene randomly inherited from either parent
- **Single-point** — random split point; prefix from parent A, suffix from parent B

**Mutation:** Gaussian perturbation — each gene independently mutated with probability `rate`, adding noise drawn from N(0, `strength`).

**Replacement:** Generational with elitism. Top `elitism_count` individuals are copied unchanged; the rest are bred via selection + crossover + mutation.

```cpp
evolve::EvolutionConfig config;
config.genome_size = 42;
config.population_size = 100;
config.mutation = {.rate = 0.1f, .strength = 0.3f};
config.elitism_count = 2;
config.tournament_size = 5;

std::mt19937 rng(42);
evolve::Population pop(config, rng);

// Evaluate fitness, then evolve
for (std::size_t i = 0; i < pop.size(); ++i) {
    pop.genome(i).set_fitness(evaluate(pop.genome(i).genes()));
}
pop.evolve(rng);
```

#### Configuration (`EvolutionConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `genome_size` | 0 | Number of float genes per individual |
| `population_size` | 100 | Individuals per generation |
| `elitism_count` | 2 | Top individuals preserved unchanged |
| `tournament_size` | 5 | Candidates sampled per selection |
| `mutation.rate` | 0.1 | Per-gene mutation probability |
| `mutation.strength` | 0.3 | Gaussian noise standard deviation |
| `init_weight_min` | -1.0 | Initial gene range lower bound |
| `init_weight_max` | 1.0 | Initial gene range upper bound |

---

### NEAT (NeuroEvolution of Augmenting Topologies)

Generic topology-evolving system templated over user-defined node properties (`Props`). Evolves both the structure (nodes, connections) and parameters of graph networks starting from minimal connectivity.

Domain-specific node initialization, crossover merging, and property mutations are decoupled from the core NEAT algorithm via a `NeatPolicy<Props>` callback struct. The consumer provides the policy; `evolve` handles the rest.

**Key features:**
- **Generic templates** — `NeatPopulation<Props>`, `NeatIndividual<Props>` work with any node property type
- **NeatPolicy** — pluggable callbacks for `init_node_props`, `merge_node_props`, `mutate_properties`, `init_output_node_props`
- **Speciation** — groups genetically similar individuals to protect innovation
- **Innovation tracking** — assigns consistent IDs to structural mutations for meaningful crossover
- **Adjusted fitness** — divides fitness by species size to prevent dominant species from taking over
- **Stagnation handling** — eliminates species that stop improving

#### Mutations

| Mutation | Rate | Description |
|----------|------|-------------|
| **Weight perturb/replace** | 0.80 | Gaussian noise (90%) or full replacement (10%) on each connection |
| **Add connection** | 0.10 | New random connection between unconnected nodes |
| **Add node** | 0.03 | Split an existing connection with a new hidden node |
| **Disable connection** | 0.02 | Disable a random enabled connection |
| **Domain-specific** | — | Bias, tau, node type, activation — provided by consumer via `NeatPolicy.mutate_properties` |

All structural mutations can be applied individually or together via the composite `mutate()` function, which also invokes `policy.mutate_properties` for domain-specific property mutations.

#### Crossover

Uses innovation numbers to align parent genomes:
- **Matching genes** (same innovation): randomly inherited from either parent
- **Disjoint/excess genes** (unique to one parent): inherited from the fitter parent
- Disabled genes have a 75% chance of being re-enabled
- Node properties are merged via `policy.merge_node_props` (consumer-defined, typically 50/50 from each parent)
- Orphaned hidden nodes (no connections) are cleaned up automatically

#### Speciation

Compatibility distance between two genomes:

```
δ = (c_excess × excess / n) + (c_disjoint × disjoint / n) + (c_weight × avg_weight_diff)
```

Individuals with δ < `compatibility_threshold` are grouped into the same species.

#### Population Lifecycle

Each call to `evolve()` runs one generation:

1. **Speciate** — assign individuals to species by genetic similarity
2. **Eliminate stagnant species** — remove species that haven't improved in `stagnation_limit` generations (keeps at least `min_species_to_keep`)
3. **Compute adjusted fitness** — `fitness / species_size`
4. **Allocate offspring** — proportional to each species' total adjusted fitness
5. **Reproduce** — within each species: elitism, then crossover (75%) or cloning (25%), then mutate
6. **Replace population**

```cpp
// Build a policy for neural-net-specific node property evolution
neuralnet::NeuralMutationConfig node_cfg;
node_cfg.bias_mutate_rate = 0.4f;
node_cfg.tau_mutate_rate  = 0.2f;
auto policy = neuralnet::make_neural_neat_policy(node_cfg);

evolve::NeatPopulationConfig config;
config.population_size = 150;
config.stagnation_limit = 15;

std::mt19937 rng(42);
evolve::NeatPopulation<neuralnet::NeuralNodeProps> pop(
    num_inputs, num_outputs, config, policy, rng);

for (auto& ind : pop.individuals()) {
    neuralnet::GraphNetwork net(ind.genome);
    ind.fitness = evaluate(net);
}
pop.evolve(rng);
```

#### Configuration (`NeatPopulationConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 150 | Individuals per generation |
| `elitism_per_species` | 1 | Top individuals preserved per species |
| `interspecies_mate_rate` | 0.001 | Probability of cross-species mating |
| `stagnation_limit` | 15 | Generations without improvement before species removal |
| `min_species_to_keep` | 2 | Floor on species count after stagnation culling |

#### Speciation Parameters (`SpeciationConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compatibility_threshold` | 3.0 | Max distance to be in the same species |
| `c_excess` | 1.0 | Weight for excess gene count |
| `c_disjoint` | 1.0 | Weight for disjoint gene count |
| `c_weight` | 0.4 | Weight for average weight difference |

#### NeatMutationConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights.weight_mutate_rate` | 0.80 | Probability of mutating each connection weight |
| `weights.weight_perturb_strength` | 0.2 | Gaussian noise std dev for weight perturbation |
| `weights.weight_replace_rate` | 0.10 | Probability of fully replacing a weight |
| `add_connection_rate` | 0.10 | Probability of adding a new connection |
| `add_node_rate` | 0.03 | Probability of splitting a connection with a new node |
| `disable_connection_rate` | 0.02 | Probability of disabling a connection |

Domain-specific mutation rates (bias, tau, node type, activation) are defined in `NeuralMutationConfig` in `neuralnet` and passed to the policy — they are not part of `NeatMutationConfig`.

## Headers

| Header | Contents |
|--------|----------|
| `evolve/genome.h` | `Genome`, `MutationConfig`, `mutate()`, `crossover_uniform()`, `crossover_single_point()` |
| `evolve/population.h` | `Population`, `EvolutionConfig` |
| `evolve/innovation.h` | `InnovationCounter` |
| `evolve/neat_policy.h` | `NeatPolicy<Props>`, `default_neat_policy()` |
| `evolve/neat_operators.h` | All NEAT mutation/crossover/speciation functions, `NeatMutationConfig`, `SpeciationConfig` |
| `evolve/neat_population.h` | `NeatPopulation<Props>`, `NeatIndividual<Props>`, `NeatPopulationConfig` |

## Dependencies

The evolve library has no dependency on `neuralnet`. It operates on generic graph genomes (`GraphGenome<Props>`) and delegates all domain-specific logic to the consumer-supplied `NeatPolicy<Props>`.
