#ifndef IPFIXPROBE_CACHE_GACONFIGURATION_HPP
#define IPFIXPROBE_CACHE_GACONFIGURATION_HPP

#include <cstdint>
#include <random>
#include "movetuple.hpp"

namespace ipxp {
class GAConfiguration{
public:
    GAConfiguration(uint32_t line_size);
    GAConfiguration(const GAConfiguration& o);
    GAConfiguration();
    GAConfiguration mutate() const;
    void read_from_file(const std::string& filename);
    void read_from_file(std::ifstream& ifs);
    void write_to_file(const std::string& filename) const;
    void write_to_file(std::ofstream& ofs) const;
    bool operator==(const GAConfiguration& o) const noexcept;
    bool operator!=(const GAConfiguration& o) const noexcept;
    GAConfiguration& operator=(const GAConfiguration& o) noexcept;
    std::tuple<uint32_t,int32_t,int32_t,int32_t,int32_t,std::vector<uint32_t>> unpack() const noexcept;
    std::string to_string() const noexcept;
    static uint32_t distance(const GAConfiguration& a,const GAConfiguration& b) noexcept;
private:
    // Pocet MoveTuple je zvolen tak, aby v prumeru kazdy MoveTuple obsahoval 4 flow
    std::vector<MoveTuple> m_moves;
    int32_t m_short_pos = 0;
    int32_t m_medium_pos = 0;
    int32_t m_long_pos = 0;
    int32_t m_never_pos = 0;
    uint32_t m_insert_pos = 0;
    uint32_t m_line_size = 0;

    std::mt19937 m_rng = std::mt19937(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> m_probability_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,1000);
    std::uniform_int_distribution<std::mt19937::result_type> m_pair_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size/4 - 1);
    std::uniform_int_distribution<std::mt19937::result_type> m_count_dist = std::uniform_int_distribution<std::mt19937::result_type>(1,m_line_size/4 );
    std::uniform_int_distribution<std::mt19937::result_type> m_insert_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size - 1);
    std::uniform_int_distribution<std::mt19937::result_type> m_offset_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size);

    void mutate_counts(float probability);
    void mutate_targets(float probability);
    void mutate_insert_pos(float probability);
    void mutate_increment(float probability);
    void mutate_counts_by_one(float probability);
    void mutate_targets_by_one(float probability);
    void fix_counts() noexcept;
    void fix_targets() noexcept;
    bool roll(double probability);
    bool is_not_valid() const noexcept;
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GACONFIGURATION_HPP
