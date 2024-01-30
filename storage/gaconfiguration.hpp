//
// Created by zaida on 26.01.2024.
//

#ifndef IPFIXPROBE_CACHE_GACONFIGURATION_HPP
#define IPFIXPROBE_CACHE_GACONFIGURATION_HPP

#include <cstdint>
#include <random>
#include "movetuple.hpp"

namespace ipxp {
class GAConfiguration{
public:
    GAConfiguration(uint32_t line_size);
    GAConfiguration();
    GAConfiguration mutate() const;
    void read_from_file(const std::string& filename);
    void write_to_file(const std::string& filename) const;
    bool operator==(const GAConfiguration& o) const noexcept;
    bool operator!=(const GAConfiguration& o) const noexcept;
    std::pair<uint32_t,std::vector<uint32_t>> unpack() const noexcept;
private:
    std::vector<MoveTuple> m_moves;
    uint16_t m_insert_pos = 0;
    uint32_t m_line_size = 0;

    std::mt19937 m_rng = std::mt19937(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> m_probability_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,1000);
    std::uniform_int_distribution<std::mt19937::result_type> m_pair_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size/4 - 1);
    std::uniform_int_distribution<std::mt19937::result_type> m_count_dist = std::uniform_int_distribution<std::mt19937::result_type>(1,m_line_size/4 );
    std::uniform_int_distribution<std::mt19937::result_type> m_insert_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size - 1);

    void mutate_counts(float probability);
    void mutate_targets(float probability);
    void mutate_insert_pos(float probability) noexcept;
    void mutate_increment(float probability);
    void fix() noexcept;
    void fix_counts() noexcept;
    void fix_targets() noexcept;
    bool roll(double probability);
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GACONFIGURATION_HPP
