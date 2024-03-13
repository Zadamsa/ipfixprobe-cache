#include "palrucache.hpp"
namespace ipxp {


alignas(32) static constexpr uint64_t masks[16][2] =
    {{0x0706050403020100, 0x0f0e0d0c0b0a0908},
     {0x0706050403020001, 0x0f0e0d0c0b0a0908},
     {0x0706050403010002, 0x0f0e0d0c0b0a0908},
     {0x0706050402010003, 0x0f0e0d0c0b0a0908},
     {0x0706050302010004, 0x0f0e0d0c0b0a0908},
     {0x0706040302010005, 0x0f0e0d0c0b0a0908},
     {0x0705040302010006, 0x0f0e0d0c0b0a0908},
     {0x0605040302010007, 0x0f0e0d0c0b0a0908},
     {0x0605040302010008, 0x0f0e0d0c0b0a0907},
     {0x0605040302010009, 0x0f0e0d0c0b0a0807},
     {0x060504030201000a, 0x0f0e0d0c0b090807},
     {0x060504030201000b, 0x0f0e0d0c0a090807},
     {0x060504030201000c, 0x0f0e0d0b0a090807},
     {0x060504030201000d, 0x0f0e0c0b0a090807},
     {0x060504030201000e, 0x0f0d0c0b0a090807},
     {0x060504030201000f, 0x0e0d0c0b0a090807}};

alignas(32) static constexpr uint64_t rotate_mask[] = {0x0706050403020100, 0x0e0d0c0b0a09080f};
alignas(32) static constexpr uint64_t and_mask[] = {0x8000800080008000, 0x8000800080008000, 0x8000800080008000, 0x8000800080008000};
alignas(32) static constexpr uint64_t or_mask[] = {0x0080008000800080, 0x0080008000800080, 0x0080008000800080, 0x0080008000800080};
alignas(32) static constexpr uint64_t first_byte_mask[] = {0xFF, 0, 0, 0};
alignas(32) static constexpr uint64_t mid_byte_mask[] = { 0, 0, 0xFF, 0};

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
    //m_offset = 6;
}

std::pair<bool, uint32_t> PALRUCache::find_existing_record(uint64_t hashval) const noexcept
{
    //return NHTFlowCache::find_existing_record(hashval);
    uint32_t begin_line = hashval & m_line_mask;
    __m256i hash_expanded = _mm256_set1_epi16((uint16_t)MetaData::HashData{(uint16_t)(hashval >> 49),1});
    __m256i cmp_res = _mm256_xor_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg, hash_expanded);

    __m128i lower128 = _mm256_extractf128_si256(cmp_res, 0);
    __m128i upper128 = _mm256_extractf128_si256(cmp_res, 1);
    __m128i min2 = _mm_minpos_epu16(lower128);
    __m128i min1 = _mm_minpos_epu16(upper128);
    if (_mm_extract_epi16(min1, 0) != 0 && _mm_extract_epi16(min2, 0) != 0){
        return std::pair{false, 0U };
    }else{
        return std::pair{true, begin_line + (_mm_extract_epi16(min1, 0) ?
                                                                         _mm_extract_epi16(min2, 1) : (8 + _mm_extract_epi16(min1, 1))) };
    }
}

uint32_t PALRUCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{

    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups++;
    m_statistics.m_lookups2++;
    m_statistics.m_hits++;
    uint64_t base_pos = flow_index - line_index;

    __m128i extended_pos = _mm_set1_epi8(base_pos);
    //__m256i list = m_metadata[1 >> m_offset].m_lru_list;
    __m128i cmp_res = _mm_cmpeq_epi8(m_metadata[flow_index >> m_offset].m_lru_list, extended_pos);
    uint16_t most_significant_bits = _mm_movemask_epi8(cmp_res);

    uint8_t current_pos =  __builtin_ffs(most_significant_bits) - 1;
    __m128i shift_mask = _mm_load_si128((__m128i*)&masks[current_pos]);
    m_metadata[flow_index >> m_offset].m_lru_list = _mm_shuffle_epi8(m_metadata[flow_index >> m_offset].m_lru_list, shift_mask);

    return flow_index;
}

std::pair<bool, uint32_t> PALRUCache::find_empty_place(uint32_t begin_line) const noexcept{
    __m256i mask = _mm256_load_si256((__m256i*)&or_mask);
    uint32_t most_significant_bits = _mm256_movemask_epi8(_mm256_or_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg,mask));
    auto res = (~most_significant_bits == 0) ? std::pair{false,0U} : std::pair{true,begin_line + (__builtin_ffs(~most_significant_bits) - 1)/2};
    return res;
}

uint32_t PALRUCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    uint8_t last_flow_index = _mm_extract_epi8(m_metadata[line_begin >> m_offset].m_lru_list,15);
    __m128i shift_mask = _mm_load_si128((__m128i*)&rotate_mask);
    m_metadata[line_begin >> m_offset].m_lru_list = _mm_shuffle_epi8(m_metadata[line_begin >> m_offset].m_lru_list, shift_mask);

    prepare_and_export(line_begin + last_flow_index, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    return line_begin + last_flow_index;
}

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