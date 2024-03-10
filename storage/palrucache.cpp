#include "palrucache.hpp"
namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("palrucache", []() { return new PALRUCache(); });
    register_plugin(&rec);
}

OptionsParser* PALRUCache::get_parser() const
{
    return new CacheOptParser();
}
std::string PALRUCache::get_name() const noexcept
{
    return "palrucache";
}

void PALRUCache::allocate_tables()
{
    NHTFlowCache::allocate_tables();
    m_metadata.resize(m_line_count);
}

void PALRUCache::init(OptionsParser& in_parser){
    NHTFlowCache::init(in_parser);
    m_offset = __builtin_ctz(m_line_mask);
}

std::pair<bool, uint32_t> PALRUCache::find_existing_record(uint64_t hashval) const noexcept
{
    uint32_t begin_line = hashval & m_line_mask;
    //__m128i metadata = _mm_load_si128((__m128i*)(&m_metadata[control_block_index].m_hashes));
    __m128i hash_expanded = _mm_set1_epi16((uint16_t)MetaData::HashData{(uint16_t)(hashval >> 49),1});
    __m128i cmp_res = _mm_cmpeq_epi16(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg, hash_expanded);
    auto mask = _mm_movemask_epi8(cmp_res);
    uint16_t pos = __builtin_ffs(mask);

    return pos ? std::pair{true, begin_line + (pos - 1)/2} : std::pair{false, 0U };
}

uint32_t PALRUCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{
    /*static const uint64_t masks[16][2] = {
        {0x0001020304050607, 0x08090A0B0C0D0E0F}, {0x0100020304050607, 0x08090A0B0C0D0E0F},
        {0x0200010304050607, 0x08090A0B0C0D0E0F}, {0x0300010204050607, 0x08090A0B0C0D0E0F},
        {0x0400010203050607, 0x08090A0B0C0D0E0F}, {0x0500010203040607, 0x08090A0B0C0D0E0F},
        {0x0600010203040507, 0x08090A0B0C0D0E0F}, {0x0700010203040506, 0x08090A0B0C0D0E0F},
        {0x0800010203040506, 0x07090A0B0C0D0E0F}, {0x0900010203040506, 0x07080A0B0C0D0E0F},
        {0x0A00010203040506, 0x0708090B0C0D0E0F}, {0x0B00010203040506, 0x0708090A0C0D0E0F},
        {0x0C00010203040506, 0x0708090A0B0D0E0F}, {0x0D00010203040506, 0x0708090A0B0C0E0F},
        {0x0E00010203040506, 0x0708090A0B0C0D0F}, {0x0F00010203040506, 0x0708090A0B0C0D0E},
    };*/
    /*static const uint64_t masks[8] = {
        0x0001020304050607, 0x0100020304050607,
        0x0200010304050607, 0x0300010204050607,
        0x0400010203050607, 0x0500010203040607,
        0x0600010203040507, 0x0700010203040506
    };*/
    constexpr static const uint64_t masks[8] = {
        0x0706050403020100, 0x0706050403020001,
        0x0706050403010002, 0x0706050402010003,
        0x0706050302010004, 0x0706040302010005,
        0x0705040302010006, 0x0605040302010007
    };
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups++;
    m_statistics.m_lookups2++;
    m_statistics.m_hits++;
    uint8_t base_pos = flow_index - line_index;

   // __m128i list = _mm_load_si128((__m128i*)&m_metadata[flow_index >> m_offset].m_lru_list);
    //__m64 list = (__m64)m_metadata[flow_index >> m_offset].m_lru_list;
    //__m64 list = _mm_set_pi64x(m_metadata[flow_index >> m_offset].m_lru_list);
    //__m64 list = m_metadata[flow_index >> m_offset].m_lru_list;
    __m64 extended_pos = _mm_set1_pi8(base_pos);
    //__m128i extended_pos = _mm_set1_epi8(base_pos);
    //__m128i cmp_res = _mm_cmpeq_epi8(list, extended_pos);
    __m64 cmp_res = _mm_cmpeq_pi8(m_metadata[flow_index >> m_offset].m_lru_list, extended_pos);
    //uint16_t significant_bits = _mm_movemask_epi8(cmp_res);
    uint8_t significant_bits = _mm_movemask_pi8(cmp_res);
    uint8_t current_pos = __builtin_ffs(significant_bits) - 1;
    //__m128i shift_mask = _mm_load_si128((__m128i*)&masks[current_pos]);
    __m64 shift_mask = (__m64)masks[current_pos];
    //list = _mm_shuffle_epi8(list,shift_mask);
    m_metadata[flow_index >> m_offset].m_lru_list = _mm_shuffle_pi8(m_metadata[flow_index >> m_offset].m_lru_list,shift_mask);
    //_mm_store_si128((__m128i*)&m_metadata[flow_index >> m_offset].m_lru_list, list);
    //m_metadata[flow_index >> m_offset].m_lru_list = list;

    /*uint64_t extracted_bits = ((list >> (4 * (15 - current_pos))) & 0xF) << 60;
    uint64_t mask = std::numeric_limits<uint64_t>::max() >> (4 * (current_pos + 1));
    uint64_t saved_bits =  list & mask;
    list = ((list >> 4) & ~mask) | extracted_bits | saved_bits;*/

    return flow_index;
}

std::pair<bool, uint32_t> PALRUCache::find_empty_place(uint32_t begin_line) const noexcept
{
    //__m128i metadata = _mm_load_si128((__m128i*)(&m_metadata[begin_line >> m_offset].m_hashes));
    __m128i mask = _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        15, 13, 11, 9, 7, 5, 3, 1);
    __m128i metadata = _mm_shuffle_epi8(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg, mask);
    uint8_t most_significant_bits = _mm_movemask_epi8(metadata);

    return most_significant_bits != std::numeric_limits<decltype(most_significant_bits)>::max() ? std::pair{true,begin_line + __builtin_ffs(~most_significant_bits) - 1} : std::pair{false, 0U};
}

uint32_t PALRUCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    //uint32_t line_end = line_begin + m_line_size;
    //__m128i list = _mm_load_si128((__m128i*)&m_metadata[line_begin >> m_offset].m_lru_list);
    //__m64 list = m_metadata[line_begin >> m_offset].m_lru_list;
    uint8_t last_flow_index = (uint64_t)m_metadata[line_begin >> m_offset].m_lru_list >> 56;
    /*__m128i mask = _mm_set_epi8(
        8, 15, 14, 13, 12, 11, 10, 9,
        7, 6, 5, 4, 3, 2, 1, 0);*/
    //__m64 mask = (__m64)0x0001020307040506;
    constexpr static const __m64 mask = (__m64)0x0605040703020100;
    //list = _mm_shuffle_epi8(list, mask);
    m_metadata[line_begin >> m_offset].m_lru_list = _mm_shuffle_pi8(  m_metadata[line_begin >> m_offset].m_lru_list , mask);
    //m_metadata[line_begin >> m_offset].m_lru_list = list;

    /*uint8_t index = list & 0xF;
    //uint64_t extracted_bits = ((list >> (4 * (15 - pos))) & 0xF) << 60;
    uint64_t mask = std::numeric_limits<uint64_t>::max() << (m_line_size - m_insert_pos);
    uint64_t saved_bits =  list & mask;
    list = ((list >> 4) & ~mask) | (index << (m_insert_pos - 1)) | saved_bits;*/
    prepare_and_export(line_begin + last_flow_index, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    //uint32_t flow_new_index = line_begin + m_insert_pos;
    //cyclic_rotate_records(flow_new_index, line_end - 1);
    return line_begin + last_flow_index;
}

void PALRUCache::prepare_and_export(uint32_t flow_index, FlowEndReason reason) noexcept
{
    uint32_t line_index = flow_index & m_line_mask;
    m_metadata[flow_index >> m_offset].m_hashes.m_hashes_array[flow_index - line_index].m_valid = 0;
    NHTFlowCache::prepare_and_export(flow_index,reason);
}

void PALRUCache::create_new_flow(uint32_t flow_index,Packet& pkt,uint64_t hashval) noexcept{
    uint32_t line_index = flow_index & m_line_mask;
    m_metadata[line_index >> m_offset].m_hashes.m_hashes_array[flow_index - line_index] = MetaData::HashData{(uint16_t)(hashval >> 49),1};
    NHTFlowCache::create_new_flow(flow_index,pkt,hashval);
}



} // namespace ipxp