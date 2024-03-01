//
// Created by zaida on 25.02.2024.
//

#ifndef CACHE_CPP_MAXHEAP_HPP
#define CACHE_CPP_MAXHEAP_HPP

#include "cstdint"
#include "flowrecord.hpp"

namespace ipxp {

class MaxHeap {
public:
    void set_line_size(uint32_t line_size) noexcept;
    void create_from(FlowRecord** flows, timeval ts) noexcept;
    FlowRecord** heapify(FlowRecord** node)noexcept;
    void fix_heap() noexcept;
    FlowRecord** bubble_up(FlowRecord** node)noexcept;
private:
    uint32_t m_line_size;
    timeval m_last_pkt_ts;
    FlowRecord** m_flows;
};

} // namespace ipxp

#endif // CACHE_CPP_MAXHEAP_HPP
