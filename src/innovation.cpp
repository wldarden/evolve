#include <evolve/innovation.h>

namespace evolve {

uint32_t InnovationCounter::get_or_create(uint32_t from_node, uint32_t to_node) {
    auto key = std::make_pair(from_node, to_node);
    auto it = current_generation_.find(key);
    if (it != current_generation_.end()) return it->second;
    auto num = next_innovation_++;
    current_generation_[key] = num;
    return num;
}

void InnovationCounter::new_generation() {
    current_generation_.clear();
}

} // namespace evolve
