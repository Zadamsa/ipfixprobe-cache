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
    return NHTFlowCache::find_existing_record(hashval);
    uint32_t begin_line = hashval & m_line_mask;
    //__m128i metadata = _mm_load_si128((__m128i*)(&m_metadata[control_block_index].m_hashes));
    __m128i hash_expanded = _mm_set1_epi16((uint16_t)MetaData::HashData{(uint16_t)(hashval >> 49),1});
    __m128i cmp_res = _mm_xor_si128(_mm_cvtsi64_si128(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg), hash_expanded);
    __m128i min = _mm_minpos_epu16(cmp_res);
    return _mm_extract_epi16(min, 0) ? std::pair{false, 0U } :std::pair{true, begin_line + _mm_extract_epi16(min, 1)};
    //__m128i cmp_res = _mm_cmpeq_epi16(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg, hash_expanded);
    //auto mask = _mm_movemask_epi8(cmp_res);
    //uint16_t pos = __builtin_ffs(mask);

    //return pos ? std::pair{true, begin_line + (pos - 1)/2} : std::pair{false, 0U };
}

static inline __m128i expand_to_uint16_t(uint64_t v) noexcept{
    uint64_t high = (((v & 0xFF)) |
                     ((v & 0xFF00)<< 8) |
                     ((v & 0xFF0000)<< 16) |
                      (v  & 0xFF000000)<< 24);
    uint64_t low = (((v  & 0xFF00000000)) >> 32 |
                    ((v  & 0xFF0000000000) >> 24) |
                    ((v  & 0xFF000000000000) >> 16) |
                     (v   & 0xFF00000000000000) >> 8);
    return _mm_set_epi64((__m64)low,(__m64)high);
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
    /*constexpr static const uint64_t masks[8] = {
        0x0706050403020100, 0x0706050403020001,
        0x0706050403010002, 0x0706050402010003,
        0x0706050302010004, 0x0706040302010005,
        0x0705040302010006, 0x0605040302010007
    };*/
    static uint64_t masks[10] = {
        0x0100070605040302, 0x8080808080808080,
        0x0302070605040100, 0x8080808080808080,
        0x0504070603020100, 0x8080808080808080,
        0x0706050403020100, 0x8080808080808080,
        0x0706050403020100, 0x8080808080808080};
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups++;
    m_statistics.m_lookups2++;
    m_statistics.m_hits++;
    uint64_t base_pos = flow_index - line_index;

   // __m128i list = _mm_load_si128((__m128i*)&m_metadata[flow_index >> m_offset].m_lru_list);
    //__m64 list = (__m64)m_metadata[flow_index >> m_offset].m_lru_list;
    //__m64 list = _mm_set_pi64x(m_metadata[flow_index >> m_offset].m_lru_list);
    //__m64 list = m_metadata[flow_index >> m_offset].m_lru_list;
    //__m64 extended_pos = _mm_set1_pi8(base_pos);
    __m128i extended_pos = _mm_set1_epi16(base_pos);
    __m128i list = _mm_cvtsi64_si128(m_metadata[flow_index >> m_offset].m_lru_list);
    //__m128i cmp_res = _mm_cmpeq_epi8(list, extended_pos);
    __m128i cmp_res = _mm_xor_si128(list, extended_pos);
    __m128i min = _mm_minpos_epu16(cmp_res);
    //__m64 cmp_res = _mm_cmpeq_pi8(m_metadata[flow_index >> m_offset].m_lru_list, extended_pos);
    //__m64 cmp_res = _m_pxor(m_metadata[flow_index >> m_offset].m_lru_list, extended_pos);

    //uint16_t significant_bits = _mm_movemask_epi8(cmp_res);
    //uint8_t significant_bits = _mm_movemask_pi8(cmp_res);
    //uint8_t current_pos = __builtin_ffs(significant_bits) - 1;
    uint8_t current_pos = 3 - _mm_extract_epi16(min, 1);
    //__m128i shift_mask = _mm_load_si128((__m128i*)&masks[current_pos]);
    //__m64 shift_mask = (__m64)masks[current_pos];
    //list = _mm_shuffle_epi8(list,shift_mask);
    //m_metadata[flow_index >> m_offset].m_lru_list = _mm_shuffle_pi8(m_metadata[flow_index >> m_offset].m_lru_list,shift_mask);
    //_mm_store_si128((__m128i*)&m_metadata[flow_index >> m_offset].m_lru_list, list);
    //m_metadata[flow_index >> m_offset].m_lru_list = list;

    //uint64_t extracted_bits = ((list >> (4 * (15 - current_pos))) & 0xF) << 60;
    uint64_t mask = std::numeric_limits<uint64_t>::max() >> (16 * (current_pos + 1));
    uint64_t saved_bits =  m_metadata[flow_index >> m_offset].m_lru_list & mask;
    m_metadata[flow_index >> m_offset].m_lru_list = ((m_metadata[flow_index >> m_offset].m_lru_list >> 16) & ~mask) | (base_pos << 48) | saved_bits;

    return flow_index;
}

std::pair<bool, uint32_t> PALRUCache::find_empty_place(uint32_t begin_line) const noexcept{
    /*uint64_t high = (((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[0] << 48) |
                     ((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[1] << 32) |
                     ((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[2] << 16) |
                     ((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[3])) & 0x8000800080008000;
    uint64_t low = (((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[4] << 48) |
                    ((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[5] << 32) |
                    ((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[6] << 16) |
                    ((uint64_t)m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[7])) & 0x8000800080008000;
    uint32_t most_significant_bits = (~(((high * 0x200040008001)>>56) | ((low * 0x200040008001) >> 60))) & 0xFF;
    auto x = begin_line + __builtin_ffs(most_significant_bits);
    return most_significant_bits == 0 ? std::pair{false,0U} : std::pair{true,begin_line + __builtin_ffs(most_significant_bits) - 1};*/
    //uint64_t high = *((uint64_t*)&m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[0]) & 0x8000800080008000;
    //uint64_t low = *((uint64_t*)&m_metadata[begin_line >> m_offset].m_hashes.m_hashes_array[4])  & 0x8000800080008000;
    //uint8_t most_significant_bits = (((high * 0x200040008001)>>60) | ((low * 0x200040008001) >> 56));
    //auto x = begin_line + __builtin_ffs(~most_significant_bits);
    uint8_t most_significant_bits = ((m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg & 0x8000800080008000) * 0x200040008001) >> 60;
    auto res = most_significant_bits == 0xF ? std::pair{false,0U} : std::pair{true,begin_line + __builtin_ffs(~most_significant_bits) - 1};
    return res;
    return most_significant_bits == 0xF ? std::pair{false,0U} : std::pair{true,begin_line + __builtin_ffs(~most_significant_bits) - 1};
    /*static const __m128i mask = _mm_set_epi64x(0x8000800080008000, 0x8000800080008000);
    __m128i metadata_xored = _mm_and_si128(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg,mask);
    __m128i min = _mm_minpos_epu16(metadata_xored);
    return _mm_extract_epi16(min, 0) ? std::pair{false,0U} : std::pair{true,begin_line + _mm_extract_epi16(min, 1)};*/
    //__m128i metadata = _mm_load_si128((__m128i*)(&m_metadata[begin_line >> m_offset].m_hashes));
    /*__m128i mask = _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1,
        15, 13, 11, 9, 7, 5, 3, 1);
    __m128i metadata = _mm_shuffle_epi8(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg, mask);
    uint8_t most_significant_bits = _mm_movemask_epi8(metadata);

    return most_significant_bits != std::numeric_limits<decltype(most_significant_bits)>::max() ? std::pair{true,begin_line + __builtin_ffs(~most_significant_bits) - 1} : std::pair{false, 0U};*/
}

uint32_t PALRUCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    //uint32_t line_end = line_begin + m_line_size;
    //__m128i list = _mm_load_si128((__m128i*)&m_metadata[line_begin >> m_offset].m_lru_list);
    //__m64 list = m_metadata[line_begin >> m_offset].m_lru_list;
    //uint8_t last_flow_index = (uint64_t)m_metadata[line_begin >> m_offset].m_lru_list >> 56;
    uint16_t last_flow_index = m_metadata[line_begin >> m_offset].m_lru_list;
    /*__m128i mask = _mm_set_epi8(
        8, 15, 14, 13, 12, 11, 10, 9,
        7, 6, 5, 4, 3, 2, 1, 0);*/
    //__m64 mask = (__m64)0x0001020307040506;
    //constexpr static const __m64 mask = (__m64)0x0605040703020100;
    //list = _mm_shuffle_epi8(list, mask);
    //m_metadata[line_begin >> m_offset].m_lru_list = _mm_shuffle_pi8(  m_metadata[line_begin >> m_offset].m_lru_list , mask);
    const uint64_t shift_mask = 0xFFFFFFFF00000000;
    m_metadata[line_begin >> m_offset].m_lru_list = (m_metadata[line_begin >> m_offset].m_lru_list & shift_mask)
        | ((m_metadata[line_begin >> m_offset].m_lru_list & 0x00000000FFFF0000) >> 16) | ((m_metadata[line_begin >> m_offset].m_lru_list & 0x000000000000FFFF) << 16);
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

/*void PALRUCache::prepare_and_export(uint32_t flow_index, FlowEndReason reason) noexcept
{
    uint32_t line_index = flow_index & m_line_mask;
    m_metadata[flow_index >> m_offset].m_hashes.m_hashes_array[flow_index - line_index].m_valid = 0;
    NHTFlowCache::prepare_and_export(flow_index,reason);
}*/

void PALRUCache::export_flow(uint32_t index){
    uint32_t line_index = index & m_line_mask;
    m_metadata[index >> m_offset].m_hashes.m_hashes_array[index - line_index].m_valid = 0;
    NHTFlowCache::export_flow(index);
}

void PALRUCache::create_new_flow(uint32_t flow_index,Packet& pkt,uint64_t hashval) noexcept{
    uint32_t line_index = flow_index & m_line_mask;
    m_metadata[line_index >> m_offset].m_hashes.m_hashes_array[flow_index - line_index] = MetaData::HashData{(uint16_t)(hashval >> 49),1};
    NHTFlowCache::create_new_flow(flow_index,pkt,hashval);
}



} // namespace ipxp