//
// Created by zaida on 26.02.2024.
//

#ifndef CACHE_CPP_LRU2QCACHE_HPP
#define CACHE_CPP_LRU2QCACHE_HPP
#include "cstdint"
#include "cache.hpp"
namespace ipxp {

class LRU2QCache : public NHTFlowCache{
protected:
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept override;
    uint32_t make_place_for_record(uint32_t line_index) noexcept override;
    void init(OptionsParser& in_parser) override;
    std::pair<bool, uint32_t> find_empty_place_in_secondary_buffer(uint32_t begin_line) const noexcept;
    uint32_t free_place_in_full_line(uint32_t line_begin) noexcept override;
    std::string get_name() const noexcept;

    uint32_t m_delimiter;
};

} // namespace ipxp

#endif // CACHE_CPP_LRU2QCACHE_HPP
