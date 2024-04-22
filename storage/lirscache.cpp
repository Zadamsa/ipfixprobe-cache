//
// Created by zaida on 28.02.2024.
//

#include "lirscache.hpp"

namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("lirscache", []() { return new LIRSCache(); });
    register_plugin(&rec);
}

std::string LIRSCache::get_name() const noexcept
{
    return "lirscache";
}

void LIRSCache::init(OptionsParser& parser){
    NHTFlowCache::init(parser);
    m_delimiter = m_line_size/16;
    m_line_count = m_cache_size/m_line_size;
    auto list_nodes = new ListNode[m_line_count * m_nodes_in_list];
    m_lists = new ListNode*[m_line_count];
    for(uint32_t i = 0; i < m_line_count; i ++)
        m_lists[i] = &list_nodes[i * m_nodes_in_list];
}

uint32_t LIRSCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{
    //m_max_heap.heapify(&m_flow_table[flow_index]);

    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;

    //LIR got hit in stack
    if (m_flow_table[flow_index]->m_reference_type == ReferenceType::LIR){
        cyclic_rotate_records(line_index + m_delimiter,flow_index);
        return  line_index + m_delimiter;
    //Resident HIR got hit in list
    }else if (flow_index < line_index + m_delimiter){
        //Rotate to end of the list
        cyclic_rotate_records(line_index, flow_index);
        auto [found,index] = find_in_main_memory(line_index,flow_index );
        //HIR is in stack too
        if (found){
            //Rotate to the top of stack
            cyclic_rotate_records(line_index + m_delimiter,index);
            m_flow_table[index]->m_reference_type = ReferenceType::LIR;
            //Release place in list
            /*m_flow_table[line_index] = m_flow_table[m_cache_size + m_qidx];
            m_flow_table[line_index]->erase();
            m_qidx = (m_qidx + 1) % m_qsize;*/
            //Move last LIR in stack to the list
            auto last_lir_index = find_last_lir_index(line_index);
            m_flow_table[last_lir_index]->m_reference_type = ReferenceType::HIR;
            //std::swap(m_flow_table[last_lir_index],m_flow_table[line_index]);
            m_flow_table[line_index] = m_flow_table[last_lir_index];
            prune_stack(line_index);
        }/*else{
            //HIR is not in main memory
            auto [found_empty_place, index_empty_place] = find_empty_place_in_main_buffer(line_index);
            if (found_empty_place){
                cyclic_rotate_records(line_index + m_delimiter, index_empty_place);
            }else{
                prepare_and_export(line_index + m_line_size - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
                cyclic_rotate_records(line_index + m_delimiter, line_index + m_line_size - 1);
            }
            m_flow_table[line_index + m_delimiter] = m_flow_table[line_index];
        }*/
        return line_index + m_delimiter;
    }else{
        //Non-resident HIR got hit
        m_flow_table[flow_index]->m_reference_type = ReferenceType::LIR;
        cyclic_rotate_records(line_index + m_delimiter,flow_index);
        auto shift_target = 0;
        if (auto [found, empty_place_index] = find_empty_place(line_index); !found || empty_place_index >= line_index + m_delimiter){
            prepare_and_export(
                line_index + m_delimiter - 1,
                FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
            shift_target = line_index + m_delimiter - 1;
        }else
            shift_target = empty_place_index;
        cyclic_rotate_records(line_index,shift_target);

        auto last_lir_index = find_last_lir_index(line_index);
        m_flow_table[last_lir_index]->m_reference_type = ReferenceType::HIR;
        //std::swap(m_flow_table[last_lir_index],m_flow_table[line_index]);
        m_flow_table[line_index] = m_flow_table[last_lir_index];
        prune_stack(line_index);
        return line_index;
    }
}

std::pair<bool,uint32_t> LIRSCache::find_in_main_memory(uint32_t line_index,uint32_t flow_in_index ) const noexcept{
    for ( uint32_t flow_index = line_index + m_delimiter; flow_index < line_index + m_line_size; flow_index++)
        if (*m_flow_table[flow_in_index] == *m_flow_table[flow_index])
            return {true, flow_index};
    return {false, 0};
}

std::pair<bool,uint32_t> LIRSCache::find_empty_place_in_main_buffer(uint32_t line_index) const noexcept{
    for ( uint32_t flow_index = line_index + m_delimiter; flow_index < line_index + m_line_size; flow_index++)
        if (m_flow_table[flow_index]->is_empty())
            return {true, flow_index};
    // Flow was not found
    return {false, 0};
}

uint32_t LIRSCache::find_last_lir_index(uint32_t line_index) const noexcept{
    uint32_t flow_index;
    for ( flow_index = line_index + m_line_size; flow_index > line_index; flow_index--)
        if (m_flow_table[flow_index - 1]->m_reference_type == LIR)
            break ;
    return flow_index - 1;
}

uint32_t LIRSCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    uint32_t line_end = line_begin + m_line_size;
    prepare_and_export(line_begin + m_delimiter - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    cyclic_rotate_records(line_begin, line_begin + m_delimiter - 1);
    return line_begin;
}

uint32_t LIRSCache::make_place_for_record(uint32_t line_index) noexcept
{
    auto [empty_place_found, flow_index] = find_empty_place(line_index);
    if (empty_place_found) {
        m_statistics.m_empty++;
    } else {
        m_statistics.m_not_empty++;
        flow_index = free_place_in_full_line(line_index);
    }
    if (flow_index < line_index + m_delimiter)
        m_flow_table[flow_index]->m_reference_type = HIR;
    else
        m_flow_table[flow_index]->m_reference_type = LIR;
    return flow_index;
}

void LIRSCache::prune_stack(uint32_t line_index) noexcept
{
//    for(uint32_t flow_index = line_index + m_delimiter; flow_index < line_index + m_line_size; flow_index--)
//        if (m_flow_table[flow_index]->is_empty())
//            continue ;
//        else if(m_flow_table[flow_index]->m_reference_type == ReferenceType::LIR)
//            break ;
//        else
//            prepare_and_export(flow_index, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    for(uint32_t flow_index = line_index + m_line_size; flow_index > line_index + m_delimiter; flow_index--)
        if (m_flow_table[flow_index - 1]->is_empty())
            continue ;
        else if(m_flow_table[flow_index - 1]->m_reference_type == ReferenceType::LIR)
            break ;
        else
            prepare_and_export(flow_index - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
}

/*std::tuple<bool,uint32_t> LIRSCache::find_and_rotate_in_list(uint64_t hashval) noexcept
{
    uint32_t begin_line = hashval & m_line_mask/m_line_size;
    //uint32_t end_line = begin_line + m_nodes_in_list;
    for (uint32_t flow_index = 0; flow_index < m_nodes_in_list; flow_index++)
        if (m_lists[begin_line][flow_index].hash == hashval)
            return {true, flow_index};
    // Flow was not found
    return {false, 0};
}*/

} // namespace ipxp