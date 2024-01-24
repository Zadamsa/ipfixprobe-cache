/**
 * \file cache.cpp
 * \brief "NewHashTable" flow cache
 * \author Martin Zadnik <zadnik@cesnet.cz>
 * \author Vaclav Bartos <bartos@cesnet.cz>
 * \author Jiri Havranek <havranek@cesnet.cz>
 * \date 2014
 * \date 2015
 * \date 2016
 */
/*
 * Copyright (C) 2014-2016 CESNET
 *
 * LICENSE TERMS
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name of the Company nor the names of its contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 *
 *
 */

#include "cache.hpp"
#include "xxhash.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ipfixprobe/ring.h>
#include <sys/time.h>

namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec
        = PluginRecord("cache", []() { return new NHTFlowCache<PRINT_FLOW_CACHE_STATS>(); });
    register_plugin(&rec);
}

OptionsParser* NHTFlowCache::get_parser() const {
    return new CacheOptParser();
}
std::string NHTFlowCache::get_name() const noexcept {
    return "cache";
}


NHTFlowCache::NHTFlowCache()
    : m_cache_size(0)
    , m_line_size(0)
    , m_line_mask(0)
    , m_line_new_idx(0)
    , m_qsize(0)
    , m_qidx(0)
    , m_timeout_idx(0)
    , m_active(0)
    , m_inactive(0)
    , m_split_biflow(false)
    , m_keylen(0)
    , m_key()
    , m_key_inv()
    , m_flow_table(nullptr)
    , m_flow_records(nullptr)
{
    test_attributes();
}

NHTFlowCache::~NHTFlowCache()
{
    print_report();
    close();
}

void NHTFlowCache::test_attributes()
{
    static_assert(
        std::is_unsigned<decltype(DEFAULT_FLOW_CACHE_SIZE)>(),
        "Static checks of default cache sizes won't properly work without unsigned type.");
    static_assert(
        bitcount<decltype(DEFAULT_FLOW_CACHE_SIZE)>(-1) > DEFAULT_FLOW_CACHE_SIZE,
        "Flow cache size is too big to fit in variable!");
    static_assert(
        bitcount<decltype(DEFAULT_FLOW_LINE_SIZE)>(-1) > DEFAULT_FLOW_LINE_SIZE,
        "Flow cache line size is too big to fit in variable!");
    static_assert(DEFAULT_FLOW_LINE_SIZE >= 1, "Flow cache line size must be at least 1!");
    static_assert(
        DEFAULT_FLOW_CACHE_SIZE >= DEFAULT_FLOW_LINE_SIZE,
        "Flow cache size must be at least cache line size!");
}

void NHTFlowCache::get_opts_from_parser(const CacheOptParser& parser)
{
    m_cache_size = parser.m_cache_size;
    m_line_size = parser.m_line_size;
    m_active = parser.m_active;
    m_inactive = parser.m_inactive;
    m_split_biflow = parser.m_split_biflow;
}
void NHTFlowCache::allocate_tables(){
    try {
        m_flow_table = std::unique_ptr<FlowRecord*[]>(new FlowRecord*[m_cache_size + m_qsize]);
        m_flow_records = std::unique_ptr<FlowRecord[]>(new FlowRecord[m_cache_size + m_qsize]);
        for (decltype(m_cache_size + m_qsize) i = 0; i < m_cache_size + m_qsize; i++) {
            m_flow_table[i] = &m_flow_records[i];
        }
    } catch (std::bad_alloc& e) {
        throw PluginError("not enough memory for flow cache allocation");
    }
}

void NHTFlowCache::init(const char* params)
{
    try {
        CacheOptParser parser;
        parser.parse(params);
        get_opts_from_parser(parser);
    } catch (ParserError& e) {
        throw PluginError(e.what());
    }

    m_line_mask = (m_cache_size - 1) & ~(m_line_size - 1);
    m_line_new_idx = m_line_size / 2;

    if (m_export_queue == nullptr) {
        throw PluginError("output queue must be set before init");
    }
    if (m_line_size > m_cache_size) {
        throw PluginError("flow cache line size must be greater or equal to cache size");
    }
    if (m_cache_size == 0) {
        throw PluginError("flow cache won't properly work with 0 records");
    }
    allocate_tables();
}

void NHTFlowCache::close()
{
    m_flow_records.reset();
    m_flow_table.reset();
}

void NHTFlowCache::set_queue(ipx_ring_t* queue)
{
    m_export_queue = queue;
    m_qsize = ipx_ring_size(queue);
}

void NHTFlowCache::export_flow(size_t index)
{
    ipx_ring_push(m_export_queue, &m_flow_table[index]->m_flow);
    std::swap(m_flow_table[index], m_flow_table[m_cache_size + m_qidx]);
    m_flow_table[index]->erase();
    m_qidx = (m_qidx + 1) % m_qsize;
}


void NHTFlowCache::finish()
{
    for (size_t m_cache_size i = 0; i < m_cache_size; i++)
        if (!m_flow_table[i]->is_empty())
            prepare_and_export(i, ipxp::FlowEndReason::FLOW_END_CACHE_SHUTDOWN);
}


void NHTFlowCache::prepare_and_export(uint32_t flow_index) noexcept
{
    plugins_pre_export(m_flow_table[flow_index]->m_flow);
    m_flow_table[flow_index]->m_flow.end_reason = get_export_reason(m_flow_table[flow_index]->m_flow);
    export_flow(flow_index);
    m_expired++;
}


void NHTFlowCache::prepare_and_export( uint32_t flow_index, uint32_t reason) noexcept{
    plugins_pre_export(m_flow_table[flow_index]->m_flow);
    m_flow_table[flow_index]->m_flow.end_reason = reason;
    export_flow(flow_index);
    m_expired++;
}

void NHTFlowCache::flush(
    Packet& pkt,
    size_t flow_index,
    int ret,
    bool source_flow)
{
    m_flushed++;
    if (ret == FLOW_FLUSH_WITH_REINSERT) {
        FlowRecord* flow = m_flow_table[flow_index];
        flow->m_flow.end_reason = FLOW_END_FORCED;
        ipx_ring_push(m_export_queue, &flow->m_flow);

        std::swap(m_flow_table[flow_index], m_flow_table[m_cache_size + m_qidx]);

        flow = m_flow_table[flow_index];
        flow->m_flow.remove_extensions();
        *flow = *m_flow_table[m_cache_size + m_qidx];
        m_qidx = (m_qidx + 1) % m_qsize;

        flow->m_flow.m_exts = nullptr;
        flow->reuse(); // Clean counters, set time first to last
        flow->update(pkt, source_flow); // Set new counters from packet

        ret = plugins_post_create(flow->m_flow, pkt);
        if (ret & FLOW_FLUSH) {
            flush(pkt, flow_index, ret, source_flow);
        }
    } else {
        m_flow_table[flow_index]->m_flow.end_reason = FLOW_END_FORCED;
        export_flow(flow_index);
    }
}

std::pair<bool, uint32_t> NHTFlowCache::find_existing_record(
    uint32_t begin_line,
    uint32_t end_line,
    uint64_t hashval) const noexcept
{
    for (uint32_t flow_index = begin_line; flow_index < end_line; flow_index++)
        if (m_flow_table[flow_index]->belongs(hashval))
            return {true, flow_index};
    return {false, 0};
}


uint32_t NHTFlowCache::enhance_existing_flow_record(
    uint32_t flow_index,
    uint32_t line_index) noexcept
{
    m_lookups += (flow_index - line_index + 1);
    m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_hits++;

    auto flow = m_flow_table[flow_index];
    for (decltype(flow_index) j = flow_index; j > line_index; j--) {
        m_flow_table[j] = m_flow_table[j - 1];
    }
    m_flow_table[line_index] = flow;
    return line_index;
}

std::pair<bool, uint32_t> NHTFlowCache::find_empty_place(
    uint32_t begin_line,
    uint32_t end_line) const noexcept
{
    for (uint32_t flow_index = begin_line; flow_index < end_line; flow_index++) {
        if (m_flow_table[flow_index]->is_empty())
            return {true, flow_index};
    }
    // No empty place was found.
    return {false, 0};
}

uint32_t NHTFlowCache::put_into_free_place(
    uint32_t flow_index,
    bool empty_place_found,
    uint32_t begin_line,
    uint32_t end_line) noexcept
{
    /* If free place was not found (flow line is full), find
     * record which will be replaced by new record. */
    if (empty_place_found)
        return flow_index;
    prepare_and_export(end_line - 1, FLOW_END_NO_RES);
    uint32_t flow_new_index = begin_line + m_line_new_idx;

    auto flow = m_flow_table[flow_index];
    for (decltype(flow_index) j = flow_index; j > flow_new_index; j--)
        m_flow_table[j] = m_flow_table[j - 1];
    m_flow_table[flow_new_index] = flow;
    return flow_new_index;
}
uint32_t NHTFlowCache::shift_records(
    uint32_t flow_index,
    uint32_t line_begin,
    uint32_t line_end) noexcept
{
    prepare_and_export(end_line - 1, FLOW_END_NO_RES);
    uint32_t flow_new_index = line_begin + m_line_new_idx;

    auto flow = m_flow_table[flow_index];
    for (decltype(flow_index) j = flow_index; j > flow_new_index; j--)
        m_flow_table[j] = m_flow_table[j - 1];
    m_flow_table[flow_new_index] = flow;
    return flow_new_index;
}

/*uint32_t NHTFlowCache<true>::put_into_free_place(
    uint32_t flow_index,
    bool empty_place_found,
    uint32_t begin_line,
    uint32_t end_line) noexcept
{
    if (empty_place_found)
        m_empty++;
    else
        m_not_empty++;
    return NHTFlowCache<false>::put_into_free_place(
        flow_index,
        empty_place_found,
        begin_line,
        end_line);
}*/

bool NHTFlowCache::tcp_connection_reset(
    Packet& pkt,
    uint32_t flow_index) noexcept
{
    uint8_t flw_flags = pkt.source_pkt ? m_flow_table[flow_index]->m_flow.src_tcp_flags
                                       : m_flow_table[flow_index]->m_flow.dst_tcp_flags;
    if ((pkt.tcp_flags & 0x02) && (flw_flags & (0x01 | 0x04))) {
        // Flows with FIN or RST TCP flags are exported when new SYN packet arrives
        m_flow_table[flow_index]->m_flow.end_reason = FLOW_END_EOF;
        export_flow(flow_index);
        put_pkt(pkt);
        return true;
    }
    return false;
}


void NHTFlowCache::create_new_flow(
    uint32_t flow_index,
    Packet& pkt,
    uint64_t hashval) noexcept
{
    m_flow_table[flow_index]->create(pkt, hashval);
    if ( plugins_post_create(m_flow_table[flow_index]->m_flow, pkt) & FLOW_FLUSH) {
        export_flow(flow_index);
        m_flushed++;
    }
}

bool NHTFlowCache::update_flow(
    uint32_t flow_index,
    Packet& pkt) noexcept
{
    auto ret = plugins_pre_update(m_flow_table[flow_index]->m_flow, pkt);
    if ( ret & FLOW_FLUSH ) {
        flush(pkt, flow_index, ret, pkt.source_pkt);
        return true;
    }
    m_flow_table[flow_index]->update(pkt, pkt.source_pkt);
    ret = plugins_post_update(m_flow_table[flow_index]->m_flow, pkt);
    if (ret & FLOW_FLUSH) {
        flush(pkt, flow_index, ret, pkt.source_pkt);
        return true;
    }
    return false;
}

std::tuple<bool,size_t,size_t,size_t,size_t> NHTFlowCache::find_flow_position(Packet& pkt) noexcept{
    /* Calculates hash value from key created before. */
    uint64_t hashval = XXH64(m_key, m_keylen, 0);
    bool source_flow = true;

    /* Get index of flow line. */
    uint32_t line_index = hashval & m_line_mask;
    uint32_t next_line = line_index + m_line_size;

    auto [found,flow_index] = find_existing_record(line_index, next_line, hashval);
    //bool  = res.first;
    //uint32_t  = res.second;

    /* Find inversed flow. */
    if (!found && !m_split_biflow) {
        uint64_t hashval_inv = XXH64(m_key_inv, m_keylen, 0);
        uint64_t line_index_inv = hashval_inv & m_line_mask;
        uint64_t next_line_inv = line_index_inv + m_line_size;
        [found,flow_index] = find_existing_record(line_index_inv, next_line_inv, hashval_inv);
        //found = res.first;
        if (found) {
            //flow_index = res.second;
            source_flow = false;
            hashval = hashval_inv;
            line_index = line_index_inv;
        }
    }
    pkt.source_pkt = source_flow;
    return {found,line_index,flow_index,next_line,hashval};

}
size_t make_place_for_record(line_index, next_line) noexcept{
    /* Existing flow record was not found. Find free place in flow line or replace some existing
         * record. */
    auto [empty_place_found,flow_index] = find_empty_place(line_index, next_line);
    //bool empty_place_found = res.first;
    //flow_index = res.second;
    if (empty_place_found){
        m_empty++;
    }else{
        m_not_empty++;
        flow_index = shift_records(flow_index,line_index,next_line);
    }
    return flow_index;
    //put_into_free_place(flow_index, empty_place_found, line_index, next_line);
}
int NHTFlowCache::insert_pkt(Packet& pkt)
{
    plugins_pre_create(pkt);

    if (!create_hash_key(pkt))
        return 0;
    auto [record_found,line_index,flow_index,next_line] = find_flow_position(pkt);
    /* Existing flow record was found, put flow record at the first index of flow line. */
    flow_index = record_found ? enhance_existing_flow_record(flow_index, line_index) : make_place_for_record(line_index, next_line);
    /*if (found)
        flow_index = enhance_existing_flow_record(flow_index, line_index);
    else
        flow_index = make_place_for_record(line_index, next_line);*/

    if (tcp_connection_reset(pkt, flow_index))
        return 0;

    if (m_flow_table[flow_index]->is_empty())
        create_new_flow(flow_index, pkt, hashval);
    else {
        /*Checks active and inactive timeouts*/
        if (timeouts_expired(pkt,flow_index))
            return insert_pkt(pkt);
        if (update_flow(flow_index, pkt))
            return 0;
    }
    export_expired(pkt.ts.tv_sec);
    return 0;
}
bool NHTFlowCache::timeouts_expired(Packet& pkt,size_t flow_index) noexcept{
    /* Check if flow record is expired (inactive timeout). */
    if (pkt.ts.tv_sec - m_flow_table[flow_index]->m_flow.time_last.tv_sec >= m_inactive) {
        prepare_and_export(flow_index, has_tcp_eof_flags(m_flow_table[flow_index]->m_flow) ? ipxp::FlowEndReason::FLOW_END_EOF : ipxp::FlowEndReason::FLOW_END_INACTIVE);
        return true;
    }
    /* Check if flow record is expired (active timeout). */
    if (pkt.ts.tv_sec - m_flow_table[flow_index]->m_flow.time_first.tv_sec >= m_active) {
        prepare_and_export(flow_index, FLOW_END_ACTIVE);
        return true;
    }
    return false;
}

int NHTFlowCache::put_pkt(Packet& pkt)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto res = insert_pkt(pkt);
    m_put_time += std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now() - start)
                      .count();
    return res;
}
static bool NHTFlowCache::has_tcp_eof_flags(const Flow& flow) noexcept{
    // When FIN or RST is set, TCP connection ended naturally
    return (flow.src_tcp_flags | flow.dst_tcp_flags) & (0x01 | 0x04);
}
/*uint8_t NHTFlowCache::get_export_reason(Flow& flow)
{
    if ((flow.src_tcp_flags | flow.dst_tcp_flags) & (0x01 | 0x04)) {

        return ipxp::FlowEndReason::FLOW_END_EOF;
    } else {
        return ipxp::FlowEndReason::FLOW_END_INACTIVE;
    }
}*/

void NHTFlowCache::export_expired(time_t ts)
{
    for (size_t m_timeout_idx i = m_timeout_idx; i < m_timeout_idx + m_line_new_idx; i++) {
        if (!m_flow_table[i]->is_empty()
            && ts - m_flow_table[i]->m_flow.time_last.tv_sec >= m_inactive) {
            prepare_and_export(i, has_tcp_eof_flags(m_flow_table[flow_index]->m_flow) ? ipxp::FlowEndReason::FLOW_END_EOF : ipxp::FlowEndReason::FLOW_END_INACTIVE);
        }
    }
    m_timeout_idx = (m_timeout_idx + m_line_new_idx) & (m_cache_size - 1);
}
// saves key value and key length into attributes NHTFlowCache::keyand NHTFlowCache::m_keylen

bool NHTFlowCache::create_hash_key(const Packet& pkt) noexcept
{
    if (pkt.ip_version == IP::v4) {
        auto key_v4 = reinterpret_cast<FlowKeyV4*>(m_key);
        auto key_v4_inv = reinterpret_cast<FlowKeyV4*>(m_key_inv);

        *key_v4 = pkt;
        key_v4_inv->save_reversed(pkt);
        m_keylen = sizeof(FlowKeyV4);
        return true;
    }
    if (pkt.ip_version == IP::v6) {
        auto key_v6 = reinterpret_cast<struct FlowKeyV6*>(m_key);
        auto key_v6_inv = reinterpret_cast<struct FlowKeyV6*>(m_key_inv);

        *key_v6 = pkt;
        key_v6_inv->save_reversed(pkt);
        m_keylen = sizeof(FlowKeyV6);
        return true;
    }
    return false;
}

void NHTFlowCache::print_report() const noexcept
{
    std::cout << m_statistics;
}

} // namespace ipxp
