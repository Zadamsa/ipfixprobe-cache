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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ipfixprobe/ring.h>
#include <sys/time.h>
#include <thread>
#include "cache.hpp"
#include "xxhash.h"
#include "flowendreason.hpp"

namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("cache", []() { return new NHTFlowCache(); });
    register_plugin(&rec);
}

OptionsParser* NHTFlowCache::get_parser() const {
    return new CacheOptParser();
}
std::string NHTFlowCache::get_name() const noexcept {
    return "cache";
}

NHTFlowCache::NHTFlowCache()
{
    test_attributes();
}

NHTFlowCache::~NHTFlowCache()
{
    m_exit = true;
    m_statistics_thread->join();
    print_report();
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
    m_periodic_statistics_sleep_time = std::chrono::duration<double>(parser.m_periodic_statistics_sleep_time);
}
void NHTFlowCache::allocate_tables(){
    try {
        m_flow_table.resize(m_cache_size + m_qsize);
        m_flow_records.resize(m_cache_size + m_qsize);
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
    m_statistics_thread = std::make_unique<std::thread>(&NHTFlowCache::export_periodic_statistics,this,std::ref(std::cout));
}

void NHTFlowCache::set_queue(ipx_ring_t* queue)
{
    m_export_queue = queue;
    m_qsize = ipx_ring_size(queue);
}

void NHTFlowCache::export_flow(uint32_t index)
{
    ipx_ring_push(m_export_queue, &m_flow_table[index]->m_flow);
    std::swap(m_flow_table[index], m_flow_table[m_cache_size + m_qidx]);
    m_flow_table[index]->erase();
    m_qidx = (m_qidx + 1) % m_qsize;
}

//Export all flows in cache on shutdown
void NHTFlowCache::finish()
{
    for (uint32_t i = 0; i < m_cache_size; i++)
        if (!m_flow_table[i]->is_empty())
            prepare_and_export(i, ipxp::FlowEndReason::FLOW_END_CACHE_SHUTDOWN);
}

void NHTFlowCache::prepare_and_export( uint32_t flow_index, FlowEndReason reason) noexcept{
    plugins_pre_export(m_flow_table[flow_index]->m_flow);
    m_flow_table[flow_index]->m_flow.end_reason = reason;
    export_flow(flow_index);
    m_statistics.m_expired++;
}

//Export flow marked by plugins on PRE_UPDATE/POST_UPDATE/POST_CREATE events
void NHTFlowCache::flush(
    Packet& pkt,
    uint32_t flow_index,
    int ret,
    bool source_flow,
    FlowEndReason reason) noexcept
{
    m_statistics.m_flushed++;
    if (ret == FLOW_FLUSH_WITH_REINSERT) {
        FlowRecord* flow = m_flow_table[flow_index];
        flow->m_flow.end_reason = reason;
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
            flush(pkt, flow_index, ret, source_flow,FlowEndReason::FLOW_END_POST_CREATE);
        }
    } else {
        m_flow_table[flow_index]->m_flow.end_reason = reason;
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
    //Flow was not found
    return {false, 0};
}

//Move flow to the first position in line
uint32_t NHTFlowCache::enhance_existing_flow_record(
    uint32_t flow_index,
    uint32_t line_index) noexcept
{
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;

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

//Export last record in line, move lower half of records down
uint32_t NHTFlowCache::shift_records(
    uint32_t line_begin,
    uint32_t line_end) noexcept
{
    prepare_and_export(line_end - 1, FlowEndReason::FLOW_END_NO_ROW_SPACE);
    uint32_t flow_new_index = line_begin + m_line_new_idx;

    auto flow = m_flow_table[line_end - 1];
    for (uint32_t j = line_end - 1; j > flow_new_index; j--)
        m_flow_table[j] = m_flow_table[j - 1];
    m_flow_table[flow_new_index] = flow;
    return flow_new_index;
}

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
        m_statistics.m_flushed++;
    }
}

//Updates flow statistics, triggers PRE_UPDATE/POST_UPDATE events
bool NHTFlowCache::update_flow(
    uint32_t flow_index,
    Packet& pkt) noexcept
{
    auto ret = plugins_pre_update(m_flow_table[flow_index]->m_flow, pkt);
    if ( ret & FLOW_FLUSH ) {
        flush(pkt, flow_index, ret, pkt.source_pkt, FlowEndReason::FLOW_END_PRE_UPDATE);
        return true;
    }
    m_flow_table[flow_index]->update(pkt, pkt.source_pkt);
    ret = plugins_post_update(m_flow_table[flow_index]->m_flow, pkt);
    if (ret & FLOW_FLUSH) {
        flush(pkt, flow_index, ret, pkt.source_pkt, FlowEndReason::FLOW_END_POST_UPDATE);
        return true;
    }
    return false;
}

std::tuple<bool,uint32_t,uint32_t,uint32_t,uint64_t> NHTFlowCache::find_flow_position(Packet& pkt) noexcept{
    /* Calculates hash value from key created before. */
    auto [ptr,size] = std::visit([](const auto& flow_key){ return std::make_pair((uint8_t*)&flow_key,sizeof(flow_key));}, m_key);
    uint64_t hashval = XXH64(ptr,size, 0);
    bool source_flow = true;
    uint32_t line_index = hashval & m_line_mask;
    uint32_t next_line = line_index + m_line_size;
    auto [found,flow_index] = find_existing_record(line_index, next_line, hashval);

    /* Find inversed flow. */
    if (!found && !m_split_biflow) {
        std::tie(ptr,size) = std::visit([](const auto& flow_key){ return std::make_pair((uint8_t*)&flow_key,sizeof(flow_key));}, m_key_inv);
        uint64_t hashval_inv = XXH64(ptr,size, 0);

        uint64_t line_index_inv = hashval_inv & m_line_mask;
        uint64_t next_line_inv = line_index_inv + m_line_size;
        std::tie(found,flow_index) = find_existing_record(line_index_inv, next_line_inv, hashval_inv);
        if (found) {
            source_flow = false;
            hashval = hashval_inv;
            line_index = line_index_inv;
        }
    }
    pkt.source_pkt = source_flow;
    return {found,line_index,flow_index,next_line,hashval};

}
// Existing flow record was not found. Find free place in flow line or replace some existing record
uint32_t NHTFlowCache::make_place_for_record(uint32_t line_index,uint32_t  next_line) noexcept{
    auto [empty_place_found,flow_index] = find_empty_place(line_index, next_line);
    if (empty_place_found){
        m_statistics.m_empty++;
    }else{
        m_statistics.m_not_empty++;
        flow_index = shift_records(line_index,next_line);
    }
    return flow_index;
}

//Main function, inserts packets to cache. Must be called via put_pkt for time measurements.
int NHTFlowCache::insert_pkt(Packet& pkt) noexcept
{
    plugins_pre_create(pkt);

    if (!create_hash_key(pkt))
        return 0;
    auto [record_found,line_index,flow_index,next_line,hashval] = find_flow_position(pkt);
    /* Existing flow record was found, put flow record at the first index of flow line. */
    flow_index = record_found ? enhance_existing_flow_record(flow_index, line_index) : make_place_for_record(line_index, next_line);
    if (tcp_connection_reset(pkt, flow_index))
        return 0;

    if (m_flow_table[flow_index]->is_empty())
        create_new_flow(flow_index, pkt, hashval);
    else {
        if (timeouts_expired(pkt,flow_index))
            return insert_pkt(pkt);
        if (update_flow(flow_index, pkt))
            return 0;
    }
    export_expired(pkt.ts.tv_sec);
    return 0;
}

//Checks active and inactive timeouts for flow, export flow if any of the timeouts expired
bool NHTFlowCache::timeouts_expired(Packet& pkt,uint32_t flow_index) noexcept{
    // Check if flow record is expired (inactive timeout)
    if (pkt.ts.tv_sec - m_flow_table[flow_index]->m_flow.time_last.tv_sec >= m_inactive) {
        prepare_and_export(flow_index, has_tcp_eof_flags(m_flow_table[flow_index]->m_flow) ? FlowEndReason::FLOW_END_TCP_EOF : FlowEndReason::FLOW_END_INACTIVE_TIMEOUT);
        return true;
    }
    // Check if flow record is expired (active timeout)
    if (pkt.ts.tv_sec - m_flow_table[flow_index]->m_flow.time_first.tv_sec >= m_active) {
        prepare_and_export(flow_index, FlowEndReason::FLOW_END_ACTIVE_TIMEOUT);
        return true;
    }
    return false;
}

//Wrapper for insert_pkt, time measurement
int NHTFlowCache::put_pkt(Packet& pkt)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto res = insert_pkt(pkt);
    m_statistics.m_put_time += std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now() - start)
                      .count();
    return res;
}
bool NHTFlowCache::has_tcp_eof_flags(const Flow& flow) noexcept{
    // When FIN or RST is set, TCP connection ended naturally
    return (flow.src_tcp_flags | flow.dst_tcp_flags) & (0x01 | 0x04);
}

// Checks if inactive timeouts expired for coherent part of table
void NHTFlowCache::export_expired(time_t ts)
{
    for (uint32_t i = m_timeout_idx; i < m_timeout_idx + m_line_new_idx; i++) {
        if (!m_flow_table[i]->is_empty() && ts - m_flow_table[i]->m_flow.time_last.tv_sec >= m_inactive) {
            prepare_and_export(i, has_tcp_eof_flags(m_flow_table[i]->m_flow) ? ipxp::FlowEndReason::FLOW_END_TCP_EOF : ipxp::FlowEndReason::FLOW_END_INACTIVE_TIMEOUT);
        }
    }
    m_timeout_idx = (m_timeout_idx + m_line_new_idx) & (m_cache_size - 1);
}

// saves key value and key length into attributes NHTFlowCache::key and NHTFlowCache::m_keylen
bool NHTFlowCache::create_hash_key(const Packet& pkt) noexcept
{
    if (pkt.ip_version != IP::v4 && pkt.ip_version != IP::v6)
        return false;
    if (pkt.ip_version == IP::v4) {
        m_key.emplace<FlowKeyV4>();
        m_key_inv.emplace<FlowKeyV4>();
    }
    if (pkt.ip_version == IP::v6) {
        m_key.emplace<FlowKeyV6>();
        m_key_inv.emplace<FlowKeyV6>();
    }
    std::visit([&pkt](auto&& flow_key){ flow_key = pkt;},m_key);
    std::visit([&pkt](auto&& flow_key){ flow_key.save_reversed(pkt);},m_key_inv);
    return true;
}

void NHTFlowCache::print_report() const noexcept
{
    if (m_statistics.m_hits){
        std::cout << "==================================================================\nTOTAL\n";
        std::cout << m_statistics;
    }
}

//Prints statistics to stream in time interval defined by NHTFlowCache::m_periodic_statistics_sleep_time. Must be called in separate thread.
void NHTFlowCache::export_periodic_statistics(std::ostream& stream) noexcept{
    while(!m_exit){
        std::this_thread::sleep_for(m_periodic_statistics_sleep_time);
        stream << m_statistics - m_last_statistics;
        m_last_statistics = m_statistics;
    }
}
} // namespace ipxp
