#ifndef CACHE_CPP_PALRUCACHE_HPP
#define CACHE_CPP_PALRUCACHE_HPP
#include "cache.hpp"
#include <emmintrin.h>
#include <mmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
namespace ipxp {
//Compilation with autoreconf -i; CPPFLAGS="-msse4.2 -mavx2 -mavx -march=native" CXXFLAGS="-msse4.2 -mavx2 -mavx -march=native" ./configure  --with-pcap; make
// Start with original cache ./ipfixprobe -i 'pcap;file=pcaps/x.pcap' -s 'cache;'
// Start with process accelerated cache ./ipfixprobe -i 'pcap;file=pcaps/x.pcap' -s 'palrucache;'
class PALRUCache :public NHTFlowCache{
public:

    std::pair<bool, uint32_t> find_existing_record(uint64_t hashval) const noexcept override;
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept override;
    std::pair<bool, uint32_t> find_empty_place(uint32_t begin_line) const noexcept override;
    uint32_t free_place_in_full_line(uint32_t line_begin) noexcept override;
    OptionsParser* get_parser() const override;
    std::string get_name() const noexcept override;
    void init(OptionsParser& in_parser) override;
    void create_new_flow(uint32_t flow_index,Packet& pkt,uint64_t hashval) noexcept;
    void allocate_tables() override;
    void export_flow(uint32_t index) override;
    void get_opts_from_parser(const CacheOptParser& parser) override;

private:
    uint16_t m_offset; ///< Offset to calculate index to m_metadata array from flow index
    struct alignas(32) MetaData {
        struct HashData {
            uint16_t m_hash : 15;
            uint16_t m_valid : 1;
            operator uint16_t () const noexcept{
                return *(uint16_t*) this;
            }
        };
        union{
            HashData m_hashes_array[16] = {}; ///< Always 16 flows in the row regardless command line options
            __m256i m_hashes_reg;
        } m_hashes;
        __m128i m_lru_list = _mm_set_epi64x(0x0f0e0d0c0b0a0908, 0x0706050403020100); ///< Indexes of flows in every byte sorted by last access time
    };
    std::vector<MetaData> m_metadata; ///< Metadata for every row of cache table
};

} // namespace ipxp

#endif // CACHE_CPP_PALRUCACHE_HPP
