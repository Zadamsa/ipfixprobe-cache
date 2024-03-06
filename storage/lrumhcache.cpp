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
    m_pkt = &pkt;

    /*if (!check_correctness()){
        std::cout<<"incorrect!\n";
        exit(666);
    }*/
    return NHTFlowCache::insert_pkt(pkt);
}

bool LRUMHCache::check_correctness() const noexcept{
    for(uint32_t line = 0; line < m_cache_size/m_line_size; line++) {
        for (uint32_t flow = 1; flow <= m_line_size; flow++){
            if (m_flow_table[line * m_line_size + flow]->is_empty())
                continue;
            auto m = m_flow_table[line * m_line_size + flow - 1];
            auto l = m_flow_table[line * m_line_size + flow * 2 - 1];
            auto r = m_flow_table[line * m_line_size + flow * 2 ];
            if (flow * 2 > m_line_size)
                continue ;
            if (!l->is_empty() && l->m_flow.time_last < m->m_flow.time_last)
                return false;
            if (!r->is_empty() && flow * 2 + 1 <= m_line_size && r->m_flow.time_last < m->m_flow.time_last)
                return false;
        }
    }
    return true;
}

uint32_t LRUMHCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    //uint32_t line_end = line_begin + m_line_size;
    //std::cout<<m_flow_table[line_begin]->m_flow.time_last.tv_usec << "\n";
    prepare_and_export(line_begin, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    //return bubble_down(line_begin);
    return line_begin;
}

uint32_t LRUMHCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{
    uint32_t line_index = flow_index & m_line_mask;
    auto x = line_index + m_line_size - flow_index;
    m_statistics.m_lookups += x;
    m_statistics.m_lookups2 += x * x;
    m_statistics.m_hits++;
    return bubble_down(flow_index);
    //return flow_index;
}

std::pair<bool, uint32_t> LRUMHCache::find_empty_place(uint32_t begin_line) const noexcept{
    uint32_t end_line = begin_line + m_line_size;
    for (uint32_t flow_index = end_line; flow_index > begin_line; flow_index--) {
        if (m_flow_table[flow_index - 1]->is_empty())
            return {true, flow_index - 1};
    }
    // No empty place was found.
    return {false, 0};
    //return empty_place_pair;
}

std::pair<bool, uint32_t> LRUMHCache::find_existing_record(uint64_t hashval) const noexcept
{
    uint32_t begin_line = hashval & m_line_mask;
    uint32_t end_line = begin_line + m_line_size;
    for (uint32_t flow_index = end_line; flow_index > begin_line; flow_index--)
        if (m_flow_table[flow_index - 1]->belongs(hashval))
            return {true, flow_index - 1};
    // Flow was not found
    return {false, 0};
}

uint32_t LRUMHCache::make_place_for_record(uint32_t line_index) noexcept{
    return bubble_down(NHTFlowCache::make_place_for_record(line_index));
    //return NHTFlowCache::make_place_for_record(line_index);
}
uint32_t LRUMHCache::bubble_down(uint32_t flow_index) noexcept{
    FlowRecord** heap = &m_flow_table[flow_index & m_line_mask] - 1;
    uint32_t swap_target_index,heap_index;
    for(heap_index = flow_index - (flow_index & m_line_mask) + 1;heap_index * 2 <= m_line_size; std::swap(heap[swap_target_index],heap[heap_index]),heap_index = swap_target_index){
        swap_target_index = [this,heap] (uint32_t l,uint32_t r) {
            if (r > m_line_size)
                return l;
            if (heap[l]->is_empty() || heap[r]->is_empty())
                return !heap[l]->is_empty() ? l : r;
            return heap[l]->m_flow.time_last < heap[r]->m_flow.time_last ? l : r;
        }(heap_index * 2, heap_index * 2 + 1);
    }
    //std::swap(heap[heap_index/2],heap[heap_index]);
    return &heap[heap_index] - m_flow_table.data();
}


} // namespace ipxp