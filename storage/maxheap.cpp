//
// Created by zaida on 25.02.2024.
//

#include "maxheap.hpp"
#include "fragmentationCache/timevalUtils.hpp"
namespace ipxp {

void MaxHeap::set_line_size(uint32_t line_size) noexcept{
    m_line_size = line_size;
}

void MaxHeap::create_from(FlowRecord** flows,timeval ts) noexcept{
    m_flows = &flows[-1];
    m_last_pkt_ts = ts;
}

void MaxHeap::fix_heap() noexcept{
    for(uint32_t i = 1; i <= m_line_size/2; i++)
        heapify(&m_flows[i]);
}
FlowRecord** MaxHeap::bubble_up(FlowRecord** node)noexcept{
    auto index = node - m_flows;
    while(index != 1 && (m_flows[index/2]->is_empty() || (m_flows[index]->m_last_second_access == timeval{-1,0} && m_flows[index/2]->m_last_second_access != timeval{-1,0} ))){
        std::swap(m_flows[index],m_flows[index/2]);
        index /= 2;
    }
    return &m_flows[index];
}

FlowRecord** MaxHeap::heapify(FlowRecord** node)noexcept{
    auto index = node - m_flows;
    /*while(index != 1 && (m_flows[index/2]->is_empty() || (m_last_pkt_ts == timeval{-1,0} && m_flows[index/2]->m_last_second_access != timeval{-1,0} ) ||
                          m_last_pkt_ts > m_flows[index/2]->m_last_second_access  )){
        std::swap(m_flows[index],m_flows[index/2]);
        index /= 2;
    }*/
    while(index * 2 <= m_line_size){
        FlowRecord** will_be_swapped;
        if (m_flows[index * 2]->is_empty() || m_flows[index * 2 + 1]->is_empty()){
            if (m_flows[index * 2]->is_empty() && m_flows[index * 2 + 1]->is_empty())
                break;
            will_be_swapped = m_flows[index * 2]->is_empty() ? &m_flows[index * 2 + 1] : &m_flows[index * 2];
        }else{
            will_be_swapped = std::max(&m_flows[index * 2 + 1], &m_flows[index * 2],[](FlowRecord** const a,FlowRecord** const b){
                return (*a)->m_last_second_access < (*b)->m_last_second_access;
            });
        }
        if (!m_flows[index]->is_empty() && m_last_pkt_ts > (*will_be_swapped)->m_last_second_access)
            break ;
        auto new_index = &m_flows[index * 2 ] == will_be_swapped ?  index * 2 : index * 2 + 1;
        std::swap(m_flows[index],*will_be_swapped);
        index = new_index;

    }
    return &m_flows[index];
}


} // namespace ipxp