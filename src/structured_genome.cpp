#include <evolve/structured_genome.h>

#include <algorithm>
#include <cassert>
#include <unordered_set>

namespace evolve {

Gene& StructuredGenome::add_gene(Gene gene) {
    if (tag_index_.count(gene.tag)) {
        throw std::runtime_error("Duplicate gene tag: " + gene.tag);
    }
    auto idx = genes_.size();
    tag_index_[gene.tag] = idx;
    genes_.push_back(std::move(gene));
    return genes_.back();
}

void StructuredGenome::add_linkage_group(LinkageGroup group) {
    for (const auto& tag : group.gene_tags) {
        if (!tag_index_.count(tag)) {
            throw std::runtime_error(
                "LinkageGroup '" + group.name + "' references unknown gene tag: " + tag);
        }
    }
    groups_.push_back(std::move(group));
}

Gene& StructuredGenome::gene(const std::string& tag) {
    auto it = tag_index_.find(tag);
    if (it == tag_index_.end()) {
        throw std::runtime_error("Gene not found: " + tag);
    }
    return genes_[it->second];
}

const Gene& StructuredGenome::gene(const std::string& tag) const {
    auto it = tag_index_.find(tag);
    if (it == tag_index_.end()) {
        throw std::runtime_error("Gene not found: " + tag);
    }
    return genes_[it->second];
}

bool StructuredGenome::has_gene(const std::string& tag) const {
    return tag_index_.count(tag) > 0;
}

float StructuredGenome::get(const std::string& tag) const {
    const auto& g = gene(tag);
    if (g.values.size() != 1) {
        throw std::runtime_error(
            "Gene '" + tag + "' has " + std::to_string(g.values.size()) +
            " values, expected 1");
    }
    return g.values[0];
}

void StructuredGenome::set(const std::string& tag, float value) {
    auto& g = gene(tag);
    if (g.values.size() != 1) {
        throw std::runtime_error(
            "Gene '" + tag + "' has " + std::to_string(g.values.size()) +
            " values, expected 1");
    }
    g.values[0] = value;
}

std::vector<float> StructuredGenome::flatten(const std::string& tag_prefix) const {
    std::vector<float> result;
    for (const auto& g : genes_) {
        if (g.tag.compare(0, tag_prefix.size(), tag_prefix) == 0) {
            result.insert(result.end(), g.values.begin(), g.values.end());
        }
    }
    return result;
}

std::vector<float> StructuredGenome::flatten_all() const {
    std::vector<float> result;
    result.reserve(total_values());
    for (const auto& g : genes_) {
        result.insert(result.end(), g.values.begin(), g.values.end());
    }
    return result;
}

std::size_t StructuredGenome::total_values() const {
    std::size_t n = 0;
    for (const auto& g : genes_) {
        n += g.values.size();
    }
    return n;
}

// --- Evolution operators ---

void mutate(StructuredGenome& genome, std::mt19937& rng) {
    std::uniform_real_distribution<float> chance(0.0f, 1.0f);

    for (auto& gene : genome.genes()) {
        if (!gene.mutation.evolvable) continue;

        std::normal_distribution<float> noise(0.0f, gene.mutation.strength);

        for (auto& val : gene.values) {
            if (chance(rng) < gene.mutation.rate) {
                val += noise(rng);
                val = std::clamp(val, gene.mutation.min_val, gene.mutation.max_val);
            }
        }
    }
}

StructuredGenome crossover(
    const StructuredGenome& a,
    const StructuredGenome& b,
    std::mt19937& rng) {

    assert(a.gene_count() == b.gene_count());

    // Build a set of which gene indices belong to each linkage group
    // and pre-decide which parent each group comes from.
    std::unordered_map<std::size_t, bool> gene_from_a;  // index -> true=A, false=B
    std::unordered_set<std::size_t> grouped_indices;

    std::uniform_int_distribution<int> coin(0, 1);

    // For each linkage group, flip one coin for the whole group
    for (const auto& group : a.linkage_groups()) {
        bool use_a = (coin(rng) == 0);

        for (const auto& tag : group.gene_tags) {
            if (a.has_gene(tag)) {
                // Find the index
                for (std::size_t i = 0; i < a.genes().size(); ++i) {
                    if (a.genes()[i].tag == tag) {
                        gene_from_a[i] = use_a;
                        grouped_indices.insert(i);
                        break;
                    }
                }
            }
        }
    }

    // Build the child
    StructuredGenome child;

    for (std::size_t i = 0; i < a.genes().size(); ++i) {
        const auto& gene_a = a.genes()[i];
        const auto& gene_b = b.genes()[i];

        assert(gene_a.tag == gene_b.tag);
        assert(gene_a.values.size() == gene_b.values.size());

        // Non-evolvable genes always come from parent A
        if (!gene_a.mutation.evolvable) {
            child.add_gene(gene_a);
            continue;
        }

        bool use_a_for_this;
        if (grouped_indices.count(i)) {
            // Part of a linkage group — decision already made
            use_a_for_this = gene_from_a[i];
        } else {
            // Ungrouped — individual coin flip per gene
            use_a_for_this = (coin(rng) == 0);
        }

        child.add_gene(use_a_for_this ? gene_a : gene_b);
    }

    // Copy linkage groups from parent A (structure is identical between parents)
    for (const auto& group : a.linkage_groups()) {
        child.add_linkage_group(group);
    }

    return child;
}

} // namespace evolve
