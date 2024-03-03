//
// Created by zaida on 03.03.2024.
//

#include "lrumhcache.hpp"
#include "fragmentationCache/timevalUtils.hpp"
namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("lrumhcache", []() { return new LRUMHCache(); });
    register_plugin(&rec);
}

std::string LRUMHCache::get_name() const noexcept
{
    return "lrumhcache";
}


int LRUMHCache::insert_pkt(Packet& pkt) noexcept{
    m_pkt_ts = pkt.ts;
    return NHTFlowCache::insert_pkt(pkt);
}

uint32_t LRUMHCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    //uint32_t line_end = line_begin + m_line_size;
    prepare_and_export(line_begin, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    return line_begin;
}

uint32_t LRUMHCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;
    return bubble_down(flow_index);
}
uint32_t LRUMHCache::make_place_for_record(uint32_t line_index) noexcept{
    return bubble_down(NHTFlowCache::make_place_for_record(line_index));
}
uint32_t LRUMHCache::bubble_down(uint32_t flow_index) noexcept{
    FlowRecord** heap = &m_flow_table[flow_index & m_line_mask] - 1;
    uint32_t swap_target_index,heap_index;
    for(heap_index = flow_index - (flow_index & m_line_mask) + 1;heap_index * 2 > m_line_size;
                std::swap(m_flow_table[swap_target_index],m_flow_table[heap_index]),heap_index = swap_target_index){
        swap_target_index = std::min(heap_index * 2, heap_index * 2 + 1, [this,heap] (uint32_t l,uint32_t r) {
            if (heap[l]->is_empty() || heap[r]->is_empty())
                return heap[l]->is_empty();
            return heap[l]->m_flow.time_last < heap[r]->m_flow.time_last;
        });
    }
    return heap_index;
}
} // namespace ipxp