#pragma once

#include <cstdint>
#include <map>
#include <utility>

namespace evolve {

class InnovationCounter {
public:
    uint32_t get_or_create(uint32_t from_node, uint32_t to_node);
    void new_generation();
    [[nodiscard]] uint32_t next_innovation() const noexcept { return next_innovation_; }

private:
    uint32_t next_innovation_ = 0;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> current_generation_;
};

} // namespace evolve
