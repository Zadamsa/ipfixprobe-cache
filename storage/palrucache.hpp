//
// Created by zaida on 08.03.2024.
//

#ifndef CACHE_CPP_PALRUCACHE_HPP
#define CACHE_CPP_PALRUCACHE_HPP
#include "cache.hpp"
#include <emmintrin.h>
#include <mmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
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
    //void prepare_and_export(uint32_t flow_index, FlowEndReason reason) noexcept override;
    void create_new_flow(uint32_t flow_index,Packet& pkt,uint64_t hashval) noexcept;
    void allocate_tables() override;
    void export_flow(uint32_t index) override;
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
        union{
            HashData m_hashes_array[32] = {};
            __m256i m_hashes_reg[2];
        } m_hashes;

        //uint64_t m_lru_list[2] = {0x0001020304050607,0x08090A0B0C0D0E0F};
        //__m64 m_lru_list = (__m64)0x0706050403020100;
        //uint64_t m_lru_list = 0x0000000100020003;
        //__m256i m_lru_list = _mm256_set_epi64x(0x18191a1b1c1d1e1f,  0x1011121314151617,0x08090a0b0c0d0e0f,0x0001020304050607);
        //__m256i m_lru_list = _mm256_set_epi64x(0x0001020304050607,0x08090a0b0c0d0e0f, 0x1011121314151617,0x18191a1b1c1d1e1f);
        __m256i m_lru_list = _mm256_set_epi64x(0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100);
        //uint64_t m_lru_list = 0x0405060700010203;
        //uint64_t m_lru_list = 0x0706050403020100;
    };
    std::vector<MetaData> m_metadata;
    //std::vector<MetaData, std::allocator<std::aligned_storage<sizeof(MetaData), alignof(MetaData)>::type>> m_metadata;
};

} // namespace ipxp

#endif // CACHE_CPP_PALRUCACHE_HPP
