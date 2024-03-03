//
// Created by zaida on 03.03.2024.
//

#ifndef CACHE_CPP_LRUMHCACHE_HPP
#define CACHE_CPP_LRUMHCACHE_HPP
#include "cache.hpp"
namespace ipxp {

class LRUMHCache : public NHTFlowCache{
    int insert_pkt(Packet& pkt) noexcept override;
    uint32_t free_place_in_full_line(uint32_t line_begin) noexcept;
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept;
    uint32_t make_place_for_record(uint32_t line_index) noexcept;
    uint32_t bubble_down(uint32_t flow_index) noexcept;
    std::string get_name() const noexcept override;
    timeval m_pkt_ts;
};

} // namespace ipxp

#endif // CACHE_CPP_LRUMHCACHE_HPP
