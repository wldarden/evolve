#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace evolve {

// ---------------------------------------------------------------------------
// Weight mutations
// ---------------------------------------------------------------------------

template <typename Props>
void mutate_weights(GraphGenome<Props>& genome,
                    const NeatWeightConfig& config,
                    std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::normal_distribution<float> perturb(0.0f, config.weight_perturb_strength);
    std::uniform_real_distribution<float> replace(-config.weight_replace_range,
                                                   config.weight_replace_range);

    for (auto& conn : genome.connections) {
        if (prob(rng) < config.weight_mutate_rate) {
            if (prob(rng) < config.weight_perturb_rate) {
                conn.weight += perturb(rng);
            } else {
                conn.weight = replace(rng);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Structural mutations
// ---------------------------------------------------------------------------

template <typename Props>
void add_connection(GraphGenome<Props>& genome,
                    InnovationCounter& counter,
                    std::mt19937& rng) {
    // Build set of existing connections (including disabled) keyed by (from, to).
    std::set<uint64_t> existing;
    for (const auto& conn : genome.connections) {
        auto key = (static_cast<uint64_t>(conn.from_node) << 32) |
                    static_cast<uint64_t>(conn.to_node);
        existing.insert(key);
    }

    // Collect all valid (from, to) pairs that don't exist yet.
    // from: any node; to: any non-input node.
    // Self-connections allowed as long as target is non-input.
    std::vector<std::pair<uint32_t, uint32_t>> candidates;
    for (const auto& from : genome.nodes) {
        for (const auto& to : genome.nodes) {
            if (to.role == NodeRole::Input) continue;
            auto key = (static_cast<uint64_t>(from.id) << 32) |
                        static_cast<uint64_t>(to.id);
            if (!existing.contains(key)) {
                candidates.emplace_back(from.id, to.id);
            }
        }
    }

    if (candidates.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, candidates.size() - 1);
    auto [from_id, to_id] = candidates[pick(rng)];

    std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
    uint32_t innov = counter.get_or_create(from_id, to_id);

    genome.connections.push_back({
        .from_node  = from_id,
        .to_node    = to_id,
        .weight     = wdist(rng),
        .enabled    = true,
        .innovation = innov,
    });
}

template <typename Props>
void add_node(GraphGenome<Props>& genome,
              InnovationCounter& counter,
              const NeatPolicy<Props>& policy,
              std::mt19937& rng) {
    // Collect indices of enabled connections.
    std::vector<std::size_t> enabled_indices;
    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        if (genome.connections[i].enabled) {
            enabled_indices.push_back(i);
        }
    }
    if (enabled_indices.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, enabled_indices.size() - 1);
    std::size_t conn_idx = enabled_indices[pick(rng)];

    // Save connection data BEFORE any push_back (which may invalidate references).
    uint32_t from_id   = genome.connections[conn_idx].from_node;
    uint32_t to_id     = genome.connections[conn_idx].to_node;
    float old_weight   = genome.connections[conn_idx].weight;

    // Disable the old connection.
    genome.connections[conn_idx].enabled = false;

    // New node gets the next available ID.
    uint32_t new_id = 0;
    for (const auto& n : genome.nodes) {
        new_id = std::max(new_id, n.id + 1);
    }

    // Create new hidden node with policy-initialised properties.
    NodeGene<Props> new_node;
    new_node.id = new_id;
    new_node.role = NodeRole::Hidden;
    policy.init_node_props(new_node.props, rng);
    genome.nodes.push_back(new_node);

    // Connection from_id -> new_id with weight 1.0
    uint32_t innov_in = counter.get_or_create(from_id, new_id);
    genome.connections.push_back({
        .from_node  = from_id,
        .to_node    = new_id,
        .weight     = 1.0f,
        .enabled    = true,
        .innovation = innov_in,
    });

    // Connection new_id -> to_id with old weight
    uint32_t innov_out = counter.get_or_create(new_id, to_id);
    genome.connections.push_back({
        .from_node  = new_id,
        .to_node    = to_id,
        .weight     = old_weight,
        .enabled    = true,
        .innovation = innov_out,
    });
}

template <typename Props>
void disable_connection(GraphGenome<Props>& genome,
                        std::mt19937& rng) {
    std::vector<std::size_t> enabled_indices;
    for (std::size_t i = 0; i < genome.connections.size(); ++i) {
        if (genome.connections[i].enabled) {
            enabled_indices.push_back(i);
        }
    }
    if (enabled_indices.empty()) return;

    std::uniform_int_distribution<std::size_t> pick(0, enabled_indices.size() - 1);
    genome.connections[enabled_indices[pick(rng)]].enabled = false;
}

// ---------------------------------------------------------------------------
// Composite mutation
// ---------------------------------------------------------------------------

template <typename Props>
void mutate(GraphGenome<Props>& genome,
            InnovationCounter& counter,
            const NeatMutationConfig& config,
            const NeatPolicy<Props>& policy,
            std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    // Weight mutations (always attempted — individual rates control per-gene).
    mutate_weights(genome, config.weights, rng);

    // Structural mutations (gated by rate).
    if (prob(rng) < config.add_connection_rate) {
        add_connection(genome, counter, rng);
    }
    if (prob(rng) < config.add_node_rate) {
        add_node(genome, counter, policy, rng);
    }
    if (prob(rng) < config.disable_connection_rate) {
        disable_connection(genome, rng);
    }

    // Domain-specific property mutations via policy callback.
    policy.mutate_properties(genome, rng);
}

// ---------------------------------------------------------------------------
// Crossover
// ---------------------------------------------------------------------------

template <typename Props>
GraphGenome<Props> crossover(const GraphGenome<Props>& fitter_parent,
                              const GraphGenome<Props>& other_parent,
                              const NeatPolicy<Props>& policy,
                              std::mt19937& rng) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    // Build maps: innovation -> connection for each parent
    std::map<uint32_t, const ConnectionGene*> fitter_conns, other_conns;
    for (const auto& c : fitter_parent.connections) fitter_conns[c.innovation] = &c;
    for (const auto& c : other_parent.connections) other_conns[c.innovation] = &c;

    // Build node maps
    std::map<uint32_t, const NodeGene<Props>*> fitter_nodes, other_nodes;
    for (const auto& n : fitter_parent.nodes) fitter_nodes[n.id] = &n;
    for (const auto& n : other_parent.nodes) other_nodes[n.id] = &n;

    GraphGenome<Props> child;
    std::set<uint32_t> child_node_ids;

    // Collect all innovation numbers
    std::set<uint32_t> all_innovations;
    for (const auto& [innov, _] : fitter_conns) all_innovations.insert(innov);
    for (const auto& [innov, _] : other_conns) all_innovations.insert(innov);

    for (auto innov : all_innovations) {
        auto fit_it = fitter_conns.find(innov);
        auto oth_it = other_conns.find(innov);

        if (fit_it != fitter_conns.end() && oth_it != other_conns.end()) {
            // Matching gene
            const auto* chosen = (prob(rng) < 0.5f) ? fit_it->second : oth_it->second;
            auto conn = *chosen;
            if (!fit_it->second->enabled || !oth_it->second->enabled) {
                conn.enabled = (prob(rng) >= 0.75f);
            }
            child.connections.push_back(conn);
            child_node_ids.insert(conn.from_node);
            child_node_ids.insert(conn.to_node);
        } else if (fit_it != fitter_conns.end()) {
            // Disjoint/excess from fitter
            child.connections.push_back(*fit_it->second);
            child_node_ids.insert(fit_it->second->from_node);
            child_node_ids.insert(fit_it->second->to_node);
        }
    }

    // Inherit nodes
    for (auto id : child_node_ids) {
        auto fit_it = fitter_nodes.find(id);
        auto oth_it = other_nodes.find(id);
        if (fit_it != fitter_nodes.end() && oth_it != other_nodes.end()) {
            // Both parents have this node — merge properties via policy.
            NodeGene<Props> child_node;
            child_node.id = id;
            child_node.role = fit_it->second->role;
            policy.merge_node_props(child_node.props,
                                    fit_it->second->props,
                                    oth_it->second->props, rng);
            child.nodes.push_back(child_node);
        } else if (fit_it != fitter_nodes.end()) {
            child.nodes.push_back(*fit_it->second);
        } else if (oth_it != other_nodes.end()) {
            child.nodes.push_back(*oth_it->second);
        }
    }

    // Ensure all input/output from fitter parent present
    for (const auto& node : fitter_parent.nodes) {
        if ((node.role == NodeRole::Input || node.role == NodeRole::Output)
            && !child_node_ids.contains(node.id)) {
            child.nodes.push_back(node);
            child_node_ids.insert(node.id);
        }
    }

    std::sort(child.nodes.begin(), child.nodes.end(),
              [](const NodeGene<Props>& a, const NodeGene<Props>& b) {
                  return a.id < b.id;
              });

    // Orphan cleanup
    std::set<uint32_t> connected_nodes;
    for (const auto& conn : child.connections) {
        if (conn.enabled) {
            connected_nodes.insert(conn.from_node);
            connected_nodes.insert(conn.to_node);
        }
    }

    std::set<uint32_t> removed_ids;
    child.nodes.erase(
        std::remove_if(child.nodes.begin(), child.nodes.end(),
            [&](const NodeGene<Props>& node) {
                if (node.role != NodeRole::Hidden) return false;
                if (!connected_nodes.contains(node.id)) {
                    removed_ids.insert(node.id);
                    return true;
                }
                return false;
            }),
        child.nodes.end());

    if (!removed_ids.empty()) {
        child.connections.erase(
            std::remove_if(child.connections.begin(), child.connections.end(),
                [&](const ConnectionGene& conn) {
                    return removed_ids.contains(conn.from_node)
                        || removed_ids.contains(conn.to_node);
                }),
            child.connections.end());
    }

    return child;
}

// ---------------------------------------------------------------------------
// Compatibility distance
// ---------------------------------------------------------------------------

template <typename Props>
float compatibility_distance(const GraphGenome<Props>& a,
                             const GraphGenome<Props>& b,
                             const SpeciationConfig& config) {
    std::map<uint32_t, float> a_weights, b_weights;
    for (const auto& c : a.connections) a_weights[c.innovation] = c.weight;
    for (const auto& c : b.connections) b_weights[c.innovation] = c.weight;

    if (a_weights.empty() && b_weights.empty()) return 0.0f;

    uint32_t a_max = a_weights.empty() ? 0 : a_weights.rbegin()->first;
    uint32_t b_max = b_weights.empty() ? 0 : b_weights.rbegin()->first;
    uint32_t shared_max = std::min(a_max, b_max);

    uint32_t excess = 0, disjoint = 0, matching = 0;
    float weight_diff_sum = 0.0f;

    std::set<uint32_t> all_innovations;
    for (const auto& [innov, _] : a_weights) all_innovations.insert(innov);
    for (const auto& [innov, _] : b_weights) all_innovations.insert(innov);

    for (auto innov : all_innovations) {
        bool in_a = a_weights.contains(innov);
        bool in_b = b_weights.contains(innov);
        if (in_a && in_b) {
            matching++;
            weight_diff_sum += std::abs(a_weights[innov] - b_weights[innov]);
        } else if (innov > shared_max) {
            excess++;
        } else {
            disjoint++;
        }
    }

    float avg_weight_diff = (matching > 0)
        ? weight_diff_sum / static_cast<float>(matching)
        : 0.0f;
    float n = static_cast<float>(std::max(a.connections.size(), b.connections.size()));
    if (a.connections.size() < 20 && b.connections.size() < 20) n = 1.0f;
    if (n < 1.0f) n = 1.0f;

    return (config.c_excess * static_cast<float>(excess) / n)
         + (config.c_disjoint * static_cast<float>(disjoint) / n)
         + (config.c_weight * avg_weight_diff);
}

// ---------------------------------------------------------------------------
// Genome factory
// ---------------------------------------------------------------------------

template <typename Props>
GraphGenome<Props> create_minimal_genome(
    std::size_t num_inputs, std::size_t num_outputs,
    const NeatPolicy<Props>& policy, std::mt19937& rng) {
    GraphGenome<Props> genome;
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);

    // Input nodes get default Props.
    for (std::size_t i = 0; i < num_inputs; ++i) {
        NodeGene<Props> node;
        node.id = static_cast<uint32_t>(i);
        node.role = NodeRole::Input;
        node.props = Props{};
        genome.nodes.push_back(node);
    }

    // Output nodes: initialise properties via policy callback.
    for (std::size_t i = 0; i < num_outputs; ++i) {
        NodeGene<Props> node;
        node.id = static_cast<uint32_t>(num_inputs + i);
        node.role = NodeRole::Output;
        policy.init_output_node_props(node.props, rng);
        genome.nodes.push_back(node);
    }

    // Fully-connected input → output with random weights, sequential innovation numbers.
    uint32_t innovation = 0;
    for (std::size_t in = 0; in < num_inputs; ++in) {
        for (std::size_t out = 0; out < num_outputs; ++out) {
            genome.connections.push_back(ConnectionGene{
                .from_node  = static_cast<uint32_t>(in),
                .to_node    = static_cast<uint32_t>(num_inputs + out),
                .weight     = weight_dist(rng),
                .enabled    = true,
                .innovation = innovation++,
            });
        }
    }

    return genome;
}

} // namespace evolve
