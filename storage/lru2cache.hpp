//
// Created by zaida on 25.02.2024.
//

#ifndef CACHE_CPP_LRU2CACHE_HPP
#define CACHE_CPP_LRU2CACHE_HPP
#include "cache.hpp"
#include "maxheap.hpp"
namespace ipxp {

class LRU2Cache : public NHTFlowCache{
protected:
    //MaxHeap m_max_heap;

    //std::tuple<bool, bool,uint32_t, uint64_t> find_flow_position(Packet& pkt) noexcept override;
    void print_rows() const noexcept;
    int insert_pkt(Packet& pkt) noexcept;
    uint32_t free_place_in_full_line(uint32_t line_begin) noexcept override;
    std::pair<bool, uint32_t> find_empty_place(uint32_t begin_line) const noexcept override;
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept override;
    //void init(OptionsParser& parser) override;
    uint32_t make_place_for_record(uint32_t line_index) noexcept override;
    std::string get_name() const noexcept;
    std::pair<bool, uint32_t> find_existing_record(uint64_t hashval) const noexcept override;
    uint32_t m_print_counter = 0;
    //uint32_t make_place_for_record(uint32_t line_index) noexcept override;
};

} // namespace ipxp

#endif // CACHE_CPP_LRU2CACHE_HPP
