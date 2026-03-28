#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace evolve {

/// Per-gene mutation configuration.
struct GeneMutationConfig {
    float rate = 0.1f;       // probability of mutation per generation
    float strength = 0.3f;   // stddev of Gaussian noise added on mutation
    float min_val = -1e9f;   // clamp floor after mutation
    float max_val = 1e9f;    // clamp ceiling after mutation
    bool evolvable = true;   // false = frozen, never mutates or crosses over
};

/// A single named gene — one or more floats that represent one concept.
/// Examples: "sensor_3_range" (1 float), "weights_layer_0" (500 floats).
struct Gene {
    std::string tag;
    std::vector<float> values;
    GeneMutationConfig mutation;

    bool operator==(const Gene& other) const {
        return tag == other.tag && values == other.values;
    }
    bool operator!=(const Gene& other) const { return !(*this == other); }
};

/// A linkage group: genes that crossover as a unit.
/// During crossover, ALL genes in a linkage group come from the same parent.
struct LinkageGroup {
    std::string name;
    std::vector<std::string> gene_tags;
};

/// A genome with named, typed genes and linkage-group-aware crossover.
class StructuredGenome {
public:
    StructuredGenome() = default;

    // --- Construction ---

    /// Add a gene. The tag must be unique within this genome.
    Gene& add_gene(Gene gene);

    /// Define a linkage group. All referenced gene tags must already exist.
    void add_linkage_group(LinkageGroup group);

    // --- Access ---

    /// Get a gene by tag.
    [[nodiscard]] Gene& gene(const std::string& tag);
    [[nodiscard]] const Gene& gene(const std::string& tag) const;

    /// Check if a gene exists.
    [[nodiscard]] bool has_gene(const std::string& tag) const;

    /// Get a single-value gene's value (convenience). Throws if gene has multiple values.
    [[nodiscard]] float get(const std::string& tag) const;

    /// Set a single-value gene's value (convenience). Throws if gene has multiple values.
    void set(const std::string& tag, float value);

    /// Access all genes in insertion order.
    [[nodiscard]] const std::vector<Gene>& genes() const noexcept { return genes_; }
    [[nodiscard]] std::vector<Gene>& genes() noexcept { return genes_; }

    /// Access all linkage groups.
    [[nodiscard]] const std::vector<LinkageGroup>& linkage_groups() const noexcept { return groups_; }

    // --- Flat extraction ---

    /// Flatten all values from genes whose tag starts with the given prefix.
    /// Returns values in gene insertion order. Useful for extracting network weights.
    [[nodiscard]] std::vector<float> flatten(const std::string& tag_prefix) const;

    /// Flatten ALL gene values in insertion order.
    [[nodiscard]] std::vector<float> flatten_all() const;

    /// Total number of float values across all genes.
    [[nodiscard]] std::size_t total_values() const;

    /// Number of genes.
    [[nodiscard]] std::size_t gene_count() const noexcept { return genes_.size(); }

private:
    std::vector<Gene> genes_;
    std::vector<LinkageGroup> groups_;
    std::unordered_map<std::string, std::size_t> tag_index_;  // tag -> index in genes_
};

// --- Evolution operators ---

/// Mutate each gene according to its own GeneMutationConfig.
/// Non-evolvable genes are skipped.
void mutate(StructuredGenome& genome, std::mt19937& rng);

/// Linkage-group-aware crossover.
/// For each LinkageGroup, flip one coin: all genes in that group come from parent A or B.
/// Ungrouped genes get individual coin flips (per-gene, not per-float).
/// Non-evolvable genes are always taken from parent A (the "primary" parent).
/// Both parents must have the same gene tags in the same order.
[[nodiscard]] StructuredGenome crossover(
    const StructuredGenome& a,
    const StructuredGenome& b,
    std::mt19937& rng);

} // namespace evolve
