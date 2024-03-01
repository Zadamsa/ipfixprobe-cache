//
// Created by zaida on 28.02.2024.
//

#ifndef CACHE_CPP_LIRSCACHE_HPP
#define CACHE_CPP_LIRSCACHE_HPP
#include "cstdint"
#include "cache.hpp"
namespace ipxp {

struct ListNode{
    uint64_t hash = 0;
};

class LIRSCache : public NHTFlowCache{
    void init(OptionsParser& parser) override;
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept override;
    std::pair<bool,uint32_t> find_empty_place_in_main_buffer(uint32_t line_index) const noexcept;
    uint32_t find_last_lir_index(uint32_t line_index) const noexcept;
    uint32_t free_place_in_full_line(uint32_t line_begin) noexcept override;
    uint32_t make_place_for_record(uint32_t line_index) noexcept override;
    void prune_stack(uint32_t line_index) noexcept;
    std::pair<bool,uint32_t> find_in_main_memory(uint32_t line_index,uint32_t  flow_in_index ) const noexcept;
    std::string get_name() const noexcept;


    uint32_t m_delimiter;
    uint32_t m_line_count;
    ListNode** m_lists;
    const uint32_t m_nodes_in_list = 4;
};

} // namespace ipxp

#endif // CACHE_CPP_LIRSCACHE_HPP
