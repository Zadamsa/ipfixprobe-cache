#include "palrucache.hpp"
namespace ipxp {
//Variables that encode shuffles for _mm_shuffle_epi8, to move the byte to the beginning of the row
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

alignas(32) static constexpr uint64_t free_place_rotate_mask[] = {0x0706050403020100, 0x0e0d0c0b0a09080f};
alignas(32) static constexpr uint64_t or_mask[] = {0x0080008000800080, 0x0080008000800080, 0x0080008000800080, 0x0080008000800080};

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

void PALRUCache::get_opts_from_parser(const CacheOptParser& parser){
    NHTFlowCache::get_opts_from_parser(parser);
    m_line_size = 16;
}

std::pair<bool, uint32_t> PALRUCache::find_existing_record(uint64_t hashval) const noexcept
{
    uint32_t begin_line = hashval & m_line_mask;
    //Load 16 copies of hashes and validity bits to the 256-bit register
    __m256i hash_expanded = _mm256_set1_epi16((uint16_t)MetaData::HashData{(uint16_t)(hashval >> 49),1});
    __m256i cmp_res = _mm256_xor_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg, hash_expanded);
    __m128i lower128 = _mm256_extractf128_si256(cmp_res, 0);
    __m128i upper128 = _mm256_extractf128_si256(cmp_res, 1);
    __m128i min2 = _mm_minpos_epu16(lower128);
    __m128i min1 = _mm_minpos_epu16(upper128);
    //If none of the registers contains uint16_t 0 as minimum, hash value was not found
    if (_mm_extract_epi16(min1, 0) != 0 && _mm_extract_epi16(min2, 0) != 0){
        return std::pair{false, 0U };
    }else{
        //If zero is found in some register, return its index to the original table
        return std::pair{true, begin_line + (_mm_extract_epi16(min1, 0) ?
                                                                         _mm_extract_epi16(min2, 1) : (8 + _mm_extract_epi16(min1, 1))) };
    }
    //Here is collisions are possible, as only 15 bits are used to compare flows. However, it must speed up the program, but it is still too slow
}

//Move record on the index flow_index to the beginning of the row
uint32_t PALRUCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{

    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups++;
    m_statistics.m_lookups2++;
    m_statistics.m_hits++;
    //Row index is the value of the byte from m_lru_list variable that encodes place of the flow in the list
    uint64_t row_index = flow_index - line_index;

    __m128i extended_row_index = _mm_set1_epi8(row_index);
    __m128i cmp_res = _mm_cmpeq_epi8(m_metadata[flow_index >> m_offset].m_lru_list, extended_row_index);
    //If the position of the flow was found, corresponding byte will be set to 0xFF, all other bytes will be set to 0
    //The only set bit will indicate current position of the flow in the list
    uint16_t most_significant_bits = _mm_movemask_epi8(cmp_res);

    uint8_t current_pos =  __builtin_ffs(most_significant_bits) - 1;
    //Depending on current position in the list, load correct shift mask to move this flow record to the beginning of the line
    __m128i shift_mask = _mm_load_si128((__m128i*)&masks[current_pos]);
    m_metadata[flow_index >> m_offset].m_lru_list = _mm_shuffle_epi8(m_metadata[flow_index >> m_offset].m_lru_list, shift_mask);

    return flow_index;
}

//Record was not found, try to find empty place in the row
std::pair<bool, uint32_t> PALRUCache::find_empty_place(uint32_t begin_line) const noexcept{
    __m256i mask = _mm256_load_si256((__m256i*)&or_mask);
    //We need to extract validity bits. Validity bit is every 16-th bit. To use with _mm256_movemask_epi8, which works with bytes, set first bit of every second byte to 1
    uint32_t most_significant_bits = _mm256_movemask_epi8(_mm256_or_si256(m_metadata[begin_line >> m_offset].m_hashes.m_hashes_reg,mask));
    //if most_significant_bits == uint32_MAX there is no empty place
    auto res = (~most_significant_bits == 0) ? std::pair{false,0U} : std::pair{true,begin_line + (__builtin_ffs(~most_significant_bits) - 1)/2};
    return res;
}

//No empty place in the row that starts at index line_begin was found. The most inactive flow mustbe exported
uint32_t PALRUCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    //Last byte indicates index of the record that wasn't accessed for the longest time
    uint8_t last_flow_index = _mm_extract_epi8(m_metadata[line_begin >> m_offset].m_lru_list,15);
    __m128i shift_mask = _mm_load_si128((__m128i*)&free_place_rotate_mask);
    //Rotate the most inactive flow to the half of the row. After this record will be exported, new flow will get this place
    m_metadata[line_begin >> m_offset].m_lru_list = _mm_shuffle_epi8(m_metadata[line_begin >> m_offset].m_lru_list, shift_mask);

    prepare_and_export(line_begin + last_flow_index, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    //Return index of the empty place
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