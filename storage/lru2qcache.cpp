//
// Created by zaida on 26.02.2024.
//

#include "lru2qcache.hpp"

namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("lru2qcache", []() { return new LRU2QCache(); });
    register_plugin(&rec);
}

std::string LRU2QCache::get_name() const noexcept
{
    return "lru2qcache";
}

uint32_t LRU2QCache::enhance_existing_flow_record(uint32_t flow_index) noexcept{
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;

    if (flow_index >= line_index + m_delimiter){
        uint32_t swap_target;
        auto [found,empty_place] = find_empty_place(line_index);
        if ( !found || empty_place >= line_index + m_delimiter){
            swap_target = line_index;
            prepare_and_export(line_index + m_delimiter - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
            cyclic_rotate_records(swap_target, line_index + m_delimiter - 1);
        }else{
            swap_target = empty_place;
        }
        std::swap(m_flow_table[swap_target],m_flow_table[flow_index]);
    }else{
        cyclic_rotate_records(line_index, flow_index);
    }
    return line_index;
}

std::pair<bool, uint32_t> LRU2QCache::find_empty_place_in_secondary_buffer(uint32_t begin_line) const noexcept{
    uint32_t end_line = begin_line + m_line_size;
    //for (uint32_t flow_index = end_line - 1; flow_index >= begin_line + m_delimiter ; flow_index--) {
    for (uint32_t flow_index = end_line - 1; flow_index >= begin_line && flow_index != std::numeric_limits<uint32_t>::max(); flow_index--) {
        if (m_flow_table[flow_index]->is_empty())
            return {true, flow_index};
    }
    // No empty place was found.
    return {false, 0};
}

uint32_t LRU2QCache::make_place_for_record(uint32_t line_index) noexcept{
    //auto [empty_place_found, flow_index] = find_empty_place_in_secondary_buffer(line_index);
    auto [empty_place_found, flow_index] = find_empty_place_in_secondary_buffer(line_index);
    if (empty_place_found) {
        m_statistics.m_empty++;
    } else {
        m_statistics.m_not_empty++;
        flow_index = free_place_in_full_line(line_index);
    }
    return flow_index;
}

uint32_t LRU2QCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    uint32_t line_end = line_begin + m_line_size;
    prepare_and_export(line_end - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    uint32_t flow_new_index = line_begin + m_delimiter;
    cyclic_rotate_records(flow_new_index, line_end - 1);
    return flow_new_index;
}

void LRU2QCache::init(OptionsParser& in_parser){
    NHTFlowCache::init(in_parser);
    m_delimiter = m_insert_pos;
    m_insert_pos /= 2;
}
} // namespace ipxp