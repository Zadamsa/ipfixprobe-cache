//
// Created by zaida on 25.02.2024.
//

#include "lru2cache.hpp"
#include "fragmentationCache/timevalUtils.hpp"
namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("lru2cache", []() { return new LRU2Cache(); });
    register_plugin(&rec);
}

std::string LRU2Cache::get_name() const noexcept
{
    return "lru2cache";
}

std::tuple<bool, bool,uint32_t, uint64_t> LRU2Cache::find_flow_position(Packet& pkt) noexcept
{
    auto [ptr, size] = std::visit(
        [](const auto& flow_key) { return std::make_pair((uint8_t*) &flow_key, sizeof(flow_key)); },
        m_key);
    //Exclude swapped flag from hashing
    uint64_t hashval = hash(ptr, size - 1);
    m_max_heap.create_from(&m_flow_table[hashval & m_line_mask],pkt.ts);
    m_max_heap.fix_heap();
    auto [found, flow_index] = find_existing_record(hashval);
    auto source_flow = !found || (std::visit([](auto&& key) { return key.swapped; }, m_key) == m_flow_table[flow_index]->m_swapped);
    return {found, source_flow, flow_index, hashval};
}
void LRU2Cache::init(OptionsParser& parser){
    NHTFlowCache::init(parser);
    m_max_heap.set_line_size(m_line_size);
}

uint32_t LRU2Cache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{
    m_max_heap.heapify(&m_flow_table[flow_index]);

    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;
    //cyclic_rotate_records(line_index, flow_index);
    return line_index;
}

std::pair<bool, uint32_t> LRU2Cache::find_empty_place(uint32_t begin_line) const noexcept
{
    uint32_t end_line = begin_line + m_line_size;
    for (uint32_t flow_index = end_line - 1; flow_index >= begin_line && flow_index != std::numeric_limits<uint32_t>::max() ; flow_index--) {
        if (m_flow_table[flow_index]->is_empty())
            return {true, flow_index};
    }
    // No empty place was found.
    return {false, 0};
}

uint32_t LRU2Cache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    //uint32_t line_end = line_begin + m_line_size;
    prepare_and_export(line_begin, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    //uint32_t flow_new_index = line_begin + m_insert_pos;
    //cyclic_rotate_records(flow_new_index, line_end - 1);
    return m_max_heap.heapify(&m_flow_table[line_begin]) - m_flow_table.data();
}

uint32_t LRU2Cache::make_place_for_record(uint32_t line_index) noexcept{
    return m_max_heap.bubble_up(&m_flow_table[NHTFlowCache::make_place_for_record(line_index)]) - m_flow_table.data();
}

int LRU2Cache::insert_pkt(Packet& pkt) noexcept {
    auto res = NHTFlowCache::insert_pkt(pkt);
    if (m_print_counter++%1000 < 5)
        print_rows();
    return res;
}
void LRU2Cache::print_rows() const noexcept{
    for(uint32_t line = 0; line < m_cache_size/m_line_size; line++) {
        std::cout<<"[";
        for (uint32_t flow = 0; flow < m_line_size; flow++){
            if (m_flow_table[line * m_line_size + flow]->is_empty())
                std::cout<<"e,";
            else if (m_flow_table[line * m_line_size + flow]->m_last_second_access == timeval{-1,0})
                std::cout<<"-1,";
            else
                std::cout<<std::to_string(m_flow_table[line * m_line_size + flow]->m_last_second_access.tv_sec) << "." <<
                    std::to_string(m_flow_table[line * m_line_size + flow]->m_last_second_access.tv_usec) + ",";
        }
        std::cout<<"]\n";
    }
    std::cout<<"==============================================================================================\n\n"<<std::endl;
}
} // namespace ipxp