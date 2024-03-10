//
// Created by zaida on 08.03.2024.
//

#ifndef CACHE_CPP_PALRUCACHE_HPP
#define CACHE_CPP_PALRUCACHE_HPP
#include "cache.hpp"
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
namespace ipxp {

class PALRUCache :public NHTFlowCache{
public:

    std::pair<bool, uint32_t> find_existing_record(uint64_t hashval) const noexcept override;
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept override;
    std::pair<bool, uint32_t> find_empty_place(uint32_t begin_line) const noexcept override;
    uint32_t free_place_in_full_line(uint32_t line_begin) noexcept override;
    OptionsParser* get_parser() const override;
    std::string get_name() const noexcept override;
    void init(OptionsParser& in_parser) override;
    void prepare_and_export(uint32_t flow_index, FlowEndReason reason) noexcept override;
    void create_new_flow(uint32_t flow_index,Packet& pkt,uint64_t hashval) noexcept;
    void allocate_tables() override;
private:
    uint16_t m_offset;
    struct MetaData {
        struct HashData {
            uint16_t m_hash : 15;
            uint16_t m_valid : 1;
            operator uint16_t () const noexcept{
                return *(uint16_t*) this;
            }
        };
        union {
            alignas(16) HashData m_hashes_array[8] = {};
            __m128i m_hashes_reg;
        } m_hashes;

        //uint64_t m_lru_list[2] = {0x0001020304050607,0x08090A0B0C0D0E0F};
        __m64 m_lru_list = (__m64)0x0706050403020100;
        //uint64_t m_lru_list = 0x0001020304050607;
        //uint64_t m_lru_list = 0x0405060700010203;
        //uint64_t m_lru_list = 0x0706050403020100;
    };
    std::vector<MetaData> m_metadata;
};

} // namespace ipxp

#endif // CACHE_CPP_PALRUCACHE_HPP
