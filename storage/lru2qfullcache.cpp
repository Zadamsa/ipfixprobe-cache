//
// Created by zaida on 26.02.2024.
//

#include "lru2qfullcache.hpp"

namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("lru2qfullcache", []() { return new LRU2QFullCache(); });
    register_plugin(&rec);
}

std::string LRU2QFullCache::get_name() const noexcept
{
    return "lru2qfullcache";
}

uint32_t LRU2QFullCache::enhance_existing_flow_record(uint32_t flow_index) noexcept{
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;
    //Flow is in main memory
    if (flow_index >= line_index + m_out_delimiter){
        cyclic_rotate_records(line_index + m_out_delimiter , flow_index);
    //Flow is in Aout memory
    }else if (flow_index >= line_index + m_in_delimiter) {
        auto [found, empty_place] = find_empty_place_in_main_buffer(line_index);
        if (!found){
            prepare_and_export(line_index + m_line_size - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
            cyclic_rotate_records(line_index + m_out_delimiter, line_index + m_line_size - 1);
            std::swap(m_flow_table[line_index + m_out_delimiter],m_flow_table[flow_index]);
        }else{
            std::swap(m_flow_table[flow_index],m_flow_table[empty_place]);
        }
    //Flow is in Ain memory
    }else{
        return flow_index;
    }
    return line_index + m_out_delimiter;
}

std::pair<bool, uint32_t> LRU2QFullCache::find_empty_place_in_main_buffer(uint32_t begin_line) const noexcept{
    uint32_t end_line = begin_line + m_line_size;
    //for (uint32_t flow_index = end_line - 1; flow_index >= begin_line + m_delimiter ; flow_index--) {
    for (uint32_t flow_index = end_line - 1; flow_index >= begin_line + m_out_delimiter; flow_index--) {
        if (m_flow_table[flow_index]->is_empty())
            return {true, flow_index};
    }
    // No empty place was found.
    return {false, 0};
}

std::pair<bool, uint32_t> LRU2QFullCache::find_empty_place_in_secondary_buffer(uint32_t begin_line) const noexcept{
    uint32_t end_line = begin_line + m_line_size;
    //for (uint32_t flow_index = end_line - 1; flow_index >= begin_line + m_delimiter ; flow_index--) {
    for (uint32_t flow_index = begin_line; flow_index < end_line; flow_index++) {
        if (m_flow_table[flow_index]->is_empty())
            return {true, flow_index};
    }
    // No empty place was found.
    return {false, 0};
}

uint32_t LRU2QFullCache::make_place_for_record(uint32_t line_index) noexcept{
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

uint32_t LRU2QFullCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    uint32_t line_end = line_begin + m_line_size;
    prepare_and_export(line_begin + m_out_delimiter - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    uint32_t flow_new_index = line_begin + m_in_delimiter;
    cyclic_rotate_records(flow_new_index, line_begin + m_out_delimiter - 1);
    std::swap(m_flow_table[flow_new_index],m_flow_table[flow_new_index - 1]);
    cyclic_rotate_records(line_begin, line_begin + m_in_delimiter - 1);
    return line_begin;
}

void LRU2QFullCache::init(OptionsParser& in_parser){
    NHTFlowCache::init(in_parser);
    m_in_delimiter = m_insert_pos / 2;
    m_out_delimiter = m_in_delimiter + m_insert_pos;
    //m_insert_pos /= 2;
}
} // namespace ipxp