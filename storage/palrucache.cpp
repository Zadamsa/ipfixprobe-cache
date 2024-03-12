#include "palrucache.hpp"
namespace ipxp {

/*alignas(32) static constexpr uint64_t masks[32][4] =
    {{0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1e1f1d1c1b1a1918},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1d1f1e1c1b1a1918},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1c1f1e1d1b1a1918},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1b1f1e1d1c1a1918},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1a1f1e1d1c1b1918},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x191f1e1d1c1b1a18},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x181f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1816151413121110, 0x171f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1817151413121110, 0x161f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1817161413121110, 0x151f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1817161513121110, 0x141f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1817161514121110, 0x131f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1817161514131110, 0x121f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1817161514131210, 0x111f1e1d1c1b1a19},
     {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1817161514131211, 0x101f1e1d1c1b1a19},
     {0x0706050403020100, 0x100e0d0c0b0a0908, 0x1817161514131211, 0x0f1f1e1d1c1b1a19},
     {0x0706050403020100, 0x100f0d0c0b0a0908, 0x1817161514131211, 0x0e1f1e1d1c1b1a19},
     {0x0706050403020100, 0x100f0e0c0b0a0908, 0x1817161514131211, 0x0d1f1e1d1c1b1a19},
     {0x0706050403020100, 0x100f0e0d0b0a0908, 0x1817161514131211, 0x0c1f1e1d1c1b1a19},
     {0x0706050403020100, 0x100f0e0d0c0a0908, 0x1817161514131211, 0x0b1f1e1d1c1b1a19},
     {0x0706050403020100, 0x100f0e0d0c0b0908, 0x1817161514131211, 0x0a1f1e1d1c1b1a19},
     {0x0706050403020100, 0x100f0e0d0c0b0a08, 0x1817161514131211, 0x091f1e1d1c1b1a19},
     {0x0706050403020100, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x081f1e1d1c1b1a19},
     {0x0806050403020100, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x071f1e1d1c1b1a19},
     {0x0807050403020100, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x061f1e1d1c1b1a19},
     {0x0807060403020100, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x051f1e1d1c1b1a19},
     {0x0807060503020100, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x041f1e1d1c1b1a19},
     {0x0807060504020100, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x031f1e1d1c1b1a19},
     {0x0807060504030100, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x021f1e1d1c1b1a19},
     {0x0807060504030200, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x011f1e1d1c1b1a19},
     {0x0807060504030201, 0x100f0e0d0c0b0a09, 0x1817161514131211, 0x001f1e1d1c1b1a19}};*/
alignas(32) static constexpr uint64_t masks[32][4] =
    {{0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0706050403020001, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0706050403010002, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0706050402010003, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0706050302010004, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0706040302010005, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0705040302010006, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0605040302010007, 0x0f0e0d0c0b0a0908, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0605040302010008, 0x0f0e0d0c0b0a0907, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x0605040302010009, 0x0f0e0d0c0b0a0807, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x060504030201000a, 0x0f0e0d0c0b090807, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x060504030201000b, 0x0f0e0d0c0a090807, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x060504030201000c, 0x0f0e0d0b0a090807, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x060504030201000d, 0x0f0e0c0b0a090807, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x060504030201000e, 0x0f0d0c0b0a090807, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x060504030201000f, 0x0e0d0c0b0a090807, 0x1716151413121110, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x17161514131211ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x17161514131210ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x17161514131110ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x17161514121110ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x17161513121110ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x17161413121110ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x17151413121110ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1e1d1c1b1a1918},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1e1d1c1b1a1917},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1e1d1c1b1a1817},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1e1d1c1b191817},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1e1d1c1a191817},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1e1d1b1a191817},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1e1c1b1a191817},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1f1d1c1b1a191817},
     {0x06050403020100ff, 0x0e0d0c0b0a090807, 0x16151413121110ff, 0x1e1d1c1b1a191817}};

alignas(32) static constexpr uint64_t rotate_mask[] = {0x0706050403020100, 0x0f0e0d0c0b0a0908, 0x161514131211101f, 0x1e1d1c1b1a191817};
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
    //m_offset = __builtin_ctz(m_line_mask);
    m_offset = 6;
}

std::pair<bool, uint32_t> PALRUCache::find_existing_record(uint64_t hashval) const noexcept
{
    //return NHTFlowCache::find_existing_record(hashval);
    /*uint32_t begin_line = hashval & m_line_mask;
    __m256i hash_expanded = _mm256_set1_epi16((uint16_t)MetaData::HashData{(uint16_t)(hashval >> 49),1});
    __m256i cmp_res1 = _mm256_xor_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg[0], hash_expanded);
    __m256i cmp_res2 = _mm256_xor_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg[1], hash_expanded);
    __m256i min1 = _mm256_min_epu16(cmp_res1);
    __m256i min2 = _mm256_min_epu16(cmp_res2);
    return _mm_extract_epi16(min1, 0) != 0 && _mm_extract_epi16(min2, 0) != 0 ? std::pair{false, 0U } :
                                                                              std::pair{true, begin_line + (_mm256_extract_epi16(min1, 0) ? _mm256_extract_epi16(min2, 1) : _mm256_extract_epi16(min1, 1)) };
*/}

uint32_t PALRUCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{

    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups++;
    m_statistics.m_lookups2++;
    m_statistics.m_hits++;
    uint64_t base_pos = flow_index - line_index;

    __m256i extended_pos = _mm256_set1_epi8(base_pos);
    //__m256i list = m_metadata[1 >> m_offset].m_lru_list;
    __m256i cmp_res = _mm256_cmpeq_epi8(m_metadata[flow_index >> m_offset].m_lru_list, extended_pos);
    uint32_t most_significant_bits = _mm256_movemask_epi8(cmp_res);

    uint8_t current_pos =  __builtin_ffs(most_significant_bits) - 1;
    __m256i shift_mask = _mm256_load_si256((__m256i*)&masks[current_pos]);
    uint8_t last_in_first_half;
    if (current_pos >= 16){
        last_in_first_half = _mm256_extract_epi8(m_metadata[flow_index >> m_offset].m_lru_list,15);
    }
    m_metadata[flow_index >> m_offset].m_lru_list = _mm256_shuffle_epi8(m_metadata[flow_index >> m_offset].m_lru_list, shift_mask);
    if (current_pos >= 16){
        __m256i byte_mask1 = _mm256_load_si256((__m256i*)&first_byte_mask);
        __m256i byte_mask2 = _mm256_load_si256((__m256i*)&mid_byte_mask);
        __m256i val1 = _mm256_and_si256(_mm256_set1_epi8(base_pos),byte_mask1);
        __m256i val2 = _mm256_and_si256(_mm256_set1_epi8(last_in_first_half),byte_mask2);
        m_metadata[flow_index >> m_offset].m_lru_list = _mm256_add_epi8(m_metadata[flow_index >> m_offset].m_lru_list,val1);
        m_metadata[flow_index >> m_offset].m_lru_list = _mm256_add_epi8(m_metadata[flow_index >> m_offset].m_lru_list,val2);
    }
    return flow_index;
}

std::pair<bool, uint32_t> PALRUCache::find_empty_place(uint32_t begin_line) const noexcept{
    __m256i mask = _mm256_load_si256((__m256i*)&and_mask);
    __m256i mask2 = _mm256_load_si256((__m256i*)&or_mask);
    auto res1 = ((uint64_t)_mm256_movemask_epi8(_mm256_or_si256(_mm256_and_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg[1],mask),mask2)));
    auto res2 = (uint64_t)_mm256_movemask_epi8(_mm256_or_si256(_mm256_and_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg[0],mask),mask2)) & 0xFFFFFFFF;
    uint64_t most_significant_bits =  res1 << 32 | res2;
    auto res = (~most_significant_bits == 0) ? std::pair{false,0U} : std::pair{true,begin_line + (__builtin_ffsll(~most_significant_bits) - 1)/2};
    return res;
}

uint32_t PALRUCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    uint8_t last_flow_index = _mm256_extract_epi8(m_metadata[line_begin >> m_offset].m_lru_list,31);
    __m256i shift_mask = _mm256_load_si256((__m256i*)&rotate_mask);
    m_metadata[line_begin >> m_offset].m_lru_list = _mm256_shuffle_epi8(m_metadata[line_begin >> m_offset].m_lru_list, shift_mask);

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