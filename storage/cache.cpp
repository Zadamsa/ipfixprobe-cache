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
 * Copyright (C) 2023 CESNET
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
#include "flowendreason.hpp"
#include "xxhash.h"
#include "flowendreason.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ipfixprobe/ring.h>
#include <sys/time.h>
#include <thread>
#include "fragmentationCache/timevalUtils.hpp"
#include <cmath>
#include "packetclock.hpp"
//#include <rte_common.h>
//#include <rte_thash.h>
namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("cache", []() { return new NHTFlowCache(); });
    register_plugin(&rec);
}

OptionsParser* NHTFlowCache::get_parser() const
{
    return new CacheOptParser();
}
std::string NHTFlowCache::get_name() const noexcept
{
    return "cache";
}

NHTFlowCache::NHTFlowCache()
    : m_cache_size(0)
    , m_line_size(0)
    , m_line_mask(0)
    , m_insert_pos(0)
    , m_qsize(0)
    , m_qidx(0)
    , m_timeout_idx(0)
    , m_active(0)
    , m_inactive(0)
    , m_split_biflow(false)
    , m_enable_fragmentation_cache(true)
    , m_exit(false)
    , m_periodic_statistics_sleep_time(0s)
    , m_fragmentation_cache(0, 0)
{
    //uint32_t key[]  = {0xDEADBEEF, 0xBAADC0DE, 0xFACEFEED, 0xDEADF00D};
    //rte_convert_rss_key(key, m_rss_key, sizeof(key));
    //set_hash_function([this](const void* ptr,uint32_t len){ return rte_softrss_be((uint32_t *)ptr, len/4, (const uint8_t*)m_rss_key);});
    set_hash_function([](const void* ptr,uint32_t len){ return XXH3_64bits(ptr, len);});
    m_graph_export.m_graph_datastream = std::ofstream("graph.data");
    m_graph_export.m_graph_new_flows_datastream = std::ofstream("graph_new_flows.data");
    m_graph_export.m_graph_cusum_datastream = std::ofstream("graph_cusum.data");
    m_graph_export.m_graph_cusum_threshold_datastream = std::ofstream("graph_cusum_threshold.data");
    test_attributes();
}

void NHTFlowCache::set_hash_function(std::function<uint64_t(const void*,uint32_t)> function) noexcept{
    m_hash_function = std::move(function);
}

uint64_t NHTFlowCache::hash(const void* ptr, uint32_t len) const noexcept{
    return m_hash_function(ptr,len);
}

/**
 * @brief Cache destructor.
 * Sets m_exit to true to signalize statistics thread to exit, waits for it.
 * Prints total collected statistics.
 */
NHTFlowCache::~NHTFlowCache()
{
    m_exit = true;
    if (m_periodic_statistics_sleep_time != 0s)
        m_statistics_thread->join();
    PacketClock::stop();
    if (m_export_thread.get())
        m_export_thread->join();
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
    m_enable_fragmentation_cache = parser.m_enable_fragmentation_cache;
}

void NHTFlowCache::init(OptionsParser& in_parser){
    auto parser = dynamic_cast<CacheOptParser*>(&in_parser);
    if (!parser)
        throw PluginError("Bad parser for NHTFlowCache");
    get_opts_from_parser(*parser);
    m_line_mask = (m_cache_size - 1) & ~(m_line_size - 1);
    m_insert_pos = m_line_size / 2;

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
    if (m_periodic_statistics_sleep_time != 0s)
        m_statistics_thread = std::make_unique<std::thread>(&NHTFlowCache::export_periodic_statistics,this,std::ref(std::cout));
    m_export_thread = std::make_unique<std::thread>(&NHTFlowCache::export_thread_function,this);
    if (m_enable_fragmentation_cache) {
        try {
            m_fragmentation_cache
                = FragmentationCache(parser->m_frag_cache_size, parser->m_frag_cache_timeout);
        } catch (std::bad_alloc& e) {
            throw PluginError("not enough memory for fragment cache allocation");
        }
    }
}


void NHTFlowCache::set_queue(ipx_ring_t* queue)
{
    m_export_queue = queue;
    m_qsize = ipx_ring_size(queue);
}

/**
 * @brief Tables allocation.
 * Allocates space for main cache table m_flow_table and m_flow_records.
 * Sets m_flow_table values as pointers to m_flow_records
 */
void NHTFlowCache::allocate_tables()
{
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
/*
 * 
 * @brief Main cache initialization.
 * @param params String from command line with options.
 * Parses and checks validity of parameters, creates tables, starts statistics thread.
 * Initializes fragmentation cache.
 */
void NHTFlowCache::init(const char* params)
{
    auto parser = std::unique_ptr<OptionsParser>(get_parser());
    try {
        parser->parse(params);
    } catch (ParserError& e) {
        throw PluginError(e.what());
    }
    init(*parser);
    //gettimeofday(&m_flood_measurement.m_last_measurement, nullptr);
}

/**
 * @brief Export flow.
 * @param index Index of flow in m_flow_table.
 * Exports flow specified by index, replaces it with previously exported flow, clears it.
 */
void NHTFlowCache::export_flow(uint32_t index)
{
    ipx_ring_push(m_export_queue, &m_flow_table[index]->m_flow);
    std::swap(m_flow_table[index], m_flow_table[m_cache_size + m_qidx]);
    m_flow_table[index]->erase();
    m_qidx = (m_qidx + 1) % m_qsize;
}

/**
 * @brief Cache devastation
 * Called on cache destruction. Exports every flow that is still in cache.
 */
void NHTFlowCache::finish()
{
    for (uint32_t i = 0; i < m_cache_size; i++)
        if (!m_flow_table[i]->is_empty()) {
            prepare_and_export(i, ipxp::FlowEndReason::FLOW_END_FORCED_END);
            m_statistics.m_exported_on_cache_end++;
        }
    print_report();
}

/*void NHTFlowCache::prepare_and_export( uint32_t flow_index, FlowEndReason reason) noexcept{
            prepare_and_export(i, ipxp::FlowEndReason::FLOW_END_FORCED_END);
}*/

void NHTFlowCache::prepare_and_export(uint32_t flow_index, FlowEndReason reason) noexcept
{
    plugins_pre_export(m_flow_table[flow_index]->m_flow);
    m_flow_table[flow_index]->m_flow.end_reason = reason;
    export_flow(flow_index);
    m_statistics.m_expired++;
}

/**
 * @brief Exports flow marked by plugins on PRE_UPDATE/POST_UPDATE/POST_CREATE events.
 * @param pkt Incoming packet.
 * @param flow_index Index of flow in m_flow_table.
 * @param ret Flags set by plugins.
 * @param source_flow True if packet comes from source device to destination.
 * @param reason Export reason.
 * Exports flow, recreates same flow and calls plugins_post_create event if FLOW_FLUSH_WITH_REINSERT
 * flag is set.
 */
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
            flush(pkt, flow_index, ret, source_flow, FlowEndReason::FLOW_END_FORCED_END);
        }
    } else {
        m_flow_table[flow_index]->m_flow.end_reason = reason;
        export_flow(flow_index);
    }
}

std::pair<bool, uint32_t> NHTFlowCache::find_existing_record(uint64_t hashval) const noexcept
{
    uint32_t begin_line = hashval & m_line_mask;
    uint32_t end_line = begin_line + m_line_size;
    for (uint32_t flow_index = begin_line; flow_index < end_line; flow_index++)
        if (m_flow_table[flow_index]->belongs(hashval))
            return {true, flow_index};
    // Flow was not found
    return {false, 0};
}

/**
 * @brief Move flow to the first position in line.
 * @param flow_index Index of flow to enhance.
 * @return Index of enhanced flow.
 */
uint32_t NHTFlowCache::enhance_existing_flow_record(uint32_t flow_index) noexcept
{
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;
    cyclic_rotate_records(line_index, flow_index);
    return line_index;
}

std::pair<bool, uint32_t> NHTFlowCache::find_empty_place(uint32_t begin_line) noexcept
{
    uint32_t end_line = begin_line + m_line_size;
    for (uint32_t flow_index = begin_line; flow_index < end_line; flow_index++) {
        if (m_flow_table[flow_index]->is_empty() || timeouts_expired(PacketClock::now_as_timeval().tv_sec,flow_index))
            return {true, flow_index};
    }
    // No empty place was found.
    return {false, 0};
}

bool NHTFlowCache::tcp_connection_reset(
    Packet& pkt,
    uint32_t flow_index,
    bool source) noexcept
{
    uint8_t flw_flags = source ? m_flow_table[flow_index]->m_flow.src_tcp_flags
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
    m_flow_table[flow_index]->create(pkt, hashval, std::visit([](auto&& key)->bool { return key.swapped; }, m_key));
    if ( plugins_post_create(m_flow_table[flow_index]->m_flow, pkt) & FLOW_FLUSH) {
        export_flow(flow_index);
        m_statistics.m_flushed++;
    }
}

/**
 * @brief Export last record in line, move lower half of records down.
 * @param line_begin Target line.
 * @return Index of insert position for new flow if row is full.
 */
uint32_t NHTFlowCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    uint32_t line_end = line_begin + m_line_size;
    prepare_and_export(line_end - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    uint32_t flow_new_index = line_begin + m_insert_pos;
    cyclic_rotate_records(flow_new_index, line_end - 1);
    return flow_new_index;
}

void NHTFlowCache::cyclic_rotate_records(uint32_t begin, uint32_t end) noexcept
{
    auto flow = m_flow_table[end];
    for (uint32_t j = end; j > begin; j--)
        m_flow_table[j] = m_flow_table[j - 1];
    m_flow_table[begin] = flow;
}

/**
 * @brief Updates flow statistics, triggers PRE_UPDATE/POST_UPDATE events.
 * @param flow_index Index of flow to update.
 * @param pkt New packet for flow_index flow.
 * @return True of updated flow was flushed, false otherwise.
 */
bool NHTFlowCache::update_flow(uint32_t flow_index, Packet& pkt, bool source) noexcept
{
    auto ret = plugins_pre_update(m_flow_table[flow_index]->m_flow, pkt);
    if (ret & FLOW_FLUSH) {
        flush(pkt, flow_index, ret, source, FlowEndReason::FLOW_END_FORCED_END);
        return true;
    }
    m_flow_table[flow_index]->update(pkt, pkt.source_pkt);
    ret = plugins_post_update(m_flow_table[flow_index]->m_flow, pkt);
    if (ret & FLOW_FLUSH) {
        flush(pkt, flow_index, ret, source, FlowEndReason::FLOW_END_FORCED_END);
        return true;
    }
    return false;
}

/**
 * @brief Looks for the index of the entry corresponding to the packet.
 * @param pkt Incoming packet.
 * @return Tuple of : True if flow was found, false otherwise, index of flow if was found, hash
 * value of flow. Calculates hash from Flow Key structure, same for structure with swapped source
 * and destination addresses and ports if first search wasn't successful.
 */
std::tuple<bool, bool,uint32_t, uint64_t> NHTFlowCache::find_flow_position(Packet& pkt) noexcept
{
    /* Calculates hash value from key created before. */
    auto [ptr, size] = std::visit(
        [](const auto& flow_key) { return std::make_pair((uint8_t*) &flow_key, sizeof(flow_key)); },
        m_key);
    //Exclude swapped flag from hashing
    uint64_t hashval = hash(ptr, size - 1);
    auto [found, flow_index] = find_existing_record(hashval);
    auto source_flow = !found || (std::visit([](auto&& key) { return key.swapped; }, m_key) == m_flow_table[flow_index]->m_swapped);
    return {found, source_flow, flow_index, hashval};
}

/**
 * @brief Find free place or replace existing record.
 * @param line_index Target line.
 * @return Index of empty flow or flow to export to free space.
 * Called when existing flow record was not found. Looks for empty place, if place wasn't found
 * makes free place by free_place_in_full_line
 */
uint32_t NHTFlowCache::make_place_for_record(uint32_t line_index) noexcept
{
    auto [empty_place_found, flow_index] = find_empty_place(line_index);
    if (empty_place_found) {
        m_statistics.m_empty++;
    } else {
        m_statistics.m_not_empty++;
        flow_index = free_place_in_full_line(line_index);
    }
    return flow_index;
}

void NHTFlowCache::try_to_fill_ports_to_fragmented_packet(Packet& packet)
{
    m_fragmentation_cache.process_packet(packet);
}

bool NHTFlowCache::is_being_flooded(const Packet& pkt) noexcept{
    if (pkt.ts - m_flood_measurement.m_last_measurement < timeval{m_flood_measurement.m_interval_length,0})
        return false;

    m_flood_measurement.m_last_measurement = pkt.ts;
    auto interval_statistics = m_statistics - m_flood_statistics;
    m_flood_statistics = m_statistics;
    m_flood_measurement.m_measurement_count++;

    auto threshold = m_flood_measurement.m_last_mean + std::max(m_flood_measurement.m_threshold * m_flood_measurement.m_deviation,m_flood_measurement.m_min);
    m_flood_measurement.m_cusum = std::max(m_flood_measurement.m_cusum + interval_statistics.m_not_empty + interval_statistics.m_empty - threshold,0.0);
    auto cusum_threshold = m_flood_measurement.m_threshold * m_flood_measurement.m_deviation;
    m_flood_measurement.m_last_mean = m_flood_measurement.m_coef * (interval_statistics.m_not_empty + interval_statistics.m_empty)
        + (1 - m_flood_measurement.m_coef) * m_flood_measurement.m_last_mean;
    int32_t forecasting_error = (interval_statistics.m_not_empty + interval_statistics.m_empty) - m_flood_measurement.m_last_mean;
    m_flood_measurement.m_error_summ += forecasting_error;
    m_flood_measurement.m_error_summ2 += (forecasting_error * forecasting_error);
    auto error_mean = m_flood_measurement.m_error_summ / m_flood_measurement.m_measurement_count;
    m_flood_measurement.m_deviation = std::sqrt((m_flood_measurement.m_error_summ2 - 2 * error_mean * m_flood_measurement.m_error_summ + error_mean * error_mean)/
                                                m_flood_measurement.m_measurement_count);

    m_graph_export.m_graph_cusum_threshold_datastream << pkt.ts.tv_sec << " " << cusum_threshold << std::endl;
    m_graph_export.m_graph_cusum_datastream << pkt.ts.tv_sec << " " << m_flood_measurement.m_cusum << std::endl;
    return m_flood_measurement.m_cusum > cusum_threshold;
}

void NHTFlowCache::export_graph_data(const Packet& pkt){
    if (pkt.ts - m_graph_export.m_last_measurement < timeval{1,0})
        return ;
    auto interval_stats = m_statistics - m_graph_export.m_last_statistics;
    m_graph_export.m_last_measurement = pkt.ts;
    m_graph_export.m_last_statistics = m_statistics;
    m_graph_export.m_graph_datastream << pkt.ts.tv_sec << " " << interval_stats.m_expired << std::endl;
    m_graph_export.m_graph_new_flows_datastream << pkt.ts.tv_sec << " " << interval_stats.m_not_empty + interval_stats.m_empty << std::endl;
    if (!m_graph_export.m_graph_datastream || !m_graph_export.m_graph_new_flows_datastream)
        throw PluginError("Cant save graph to file");
}

/**
 * @brief Main packet insertion function.
 * @param pkt Incoming packet.
 * Must be called via put_pkt for time measurements.
 */
int NHTFlowCache::insert_pkt(Packet& pkt) noexcept
{
    PacketClock::set_time(pkt.ts);
    // Calls PRE_CREATE event for new packet
    plugins_pre_create(pkt);
    // Tries to fill up ports if packet is fragmented
   //if (m_enable_fragmentation_cache)
    //    try_to_fill_ports_to_fragmented_packet(pkt);
    // Saves key fields of flow to FlowKey structures
    if (!create_hash_key(pkt))
        return 0;
    export_graph_data(pkt);
    if (is_being_flooded(pkt)){
        auto raw_time = pkt.ts.tv_sec;
        tm* time_info = localtime(&raw_time);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", time_info);
        std::cout<<"Flood detected at " << buffer << "\n";
    }
    // Tries to find index of flow to which packet belongs
    auto [record_found, source, flow_index, hashval] = find_flow_position(pkt);

    bool updated;
    do{
        auto current_value = m_locked_lines.load();
        if (current_value.m_export_line == ( (uint32_t)( hashval & m_line_mask )))
            continue;
        updated = m_locked_lines.compare_exchange_weak(current_value,{ current_value.m_export_line, (uint32_t)(hashval & m_line_mask)});
    }while(!updated);


    flow_index = record_found ? enhance_existing_flow_record(flow_index)
                              : make_place_for_record(hashval & m_line_mask);
    // Reinsert flow on tcp FIN/RST flags
    if (tcp_connection_reset(pkt, flow_index,source))
        return 0;

    if (m_flow_table[flow_index]->is_empty())
        // Returned index contains no flow, so new flow can be created
        create_new_flow(flow_index, pkt, hashval);
    else {
        // Returned index contains target flow, checks for possible timeouts to reinsert
        if (timeouts_expired(pkt.ts.tv_sec, flow_index))
            return insert_pkt(pkt);
        // Finally update flow data
        if (update_flow(flow_index, pkt,source))
            return 0;
    }
    // Checks part of cache for possible timeouts
    //export_expired(pkt.ts.tv_sec);
    return 0;
}

/**
 * @brief Checks for active and inactive timeouts of the flow.
 * @param pkt Incoming packet.
 * @param flow_index Flow index
 * @return True if successfully exported flow
 * Export flow if any of the timeouts expired
 */
bool NHTFlowCache::timeouts_expired(time_t tv, uint32_t flow_index) noexcept
{
    // Check if flow record is expired (inactive timeout)
    if (tv - m_flow_table[flow_index]->m_flow.time_last.tv_sec >= m_inactive) {
        prepare_and_export(
            flow_index,
            has_tcp_eof_flags(m_flow_table[flow_index]->m_flow)
                ? FlowEndReason::FLOW_END_EOF_DETECTED
                : FlowEndReason::FLOW_END_IDLE_TIMEOUT);
        return true;
    }
    // Check if flow record is expired (active timeout)
    if (tv - m_flow_table[flow_index]->m_flow.time_first.tv_sec >= m_active) {
        prepare_and_export(flow_index, FlowEndReason::FLOW_END_ACTIVE_TIMEOUT);
        return true;
    }
    return false;
}

/**
 * @brief Time measurement for insert_pkt.
 * @param pkt Incoming packet.
 */
int NHTFlowCache::put_pkt(Packet& pkt)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto res = insert_pkt(pkt);

    while(true){
        auto current_value = m_locked_lines.load();
        if(m_locked_lines.compare_exchange_weak(current_value,{ current_value.m_export_line, (uint32_t)-1}))
            break;
    }

    m_statistics.m_put_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::high_resolution_clock::now() - start)
                      .count();
    return res;
}

bool NHTFlowCache::has_tcp_eof_flags(const Flow& flow) noexcept
{
    // When FIN or RST is set, TCP connection ended naturally
    return (flow.src_tcp_flags | flow.dst_tcp_flags) & (0x01 | 0x04);
}


/**
 * @brief Checks compartment for timeouts.
 * @param ts Timestamp of the last incoming packet.
 * Checks if inactive timeouts expired for coherent part of table
 */
void NHTFlowCache::export_expired(time_t ts)
{
    bool updated;
    do{
        AtomicLockedLine expected_value{(uint32_t)-1, (uint32_t)-1};
        updated = m_locked_lines.compare_exchange_weak(expected_value,{ m_timeout_idx & m_line_mask, (uint32_t)-1});
        if (!updated){
            if (expected_value.m_export_line != (uint32_t)-1){
                std::this_thread::yield();
            }else{
                if (expected_value.m_process_line != ( m_timeout_idx & m_line_mask ))
                    updated = m_locked_lines.compare_exchange_weak(expected_value,{ m_timeout_idx & m_line_mask, expected_value.m_process_line});
                else{
                    updated = m_locked_lines.compare_exchange_weak(expected_value,{ (m_timeout_idx + m_line_size) & m_line_mask, expected_value.m_process_line});
                    if (updated)
                        m_timeout_idx = (m_timeout_idx + m_line_size) & m_line_mask;
                }
            }
        }else
            std::cout<<"";
    }while(!updated);
    export_expired_body(ts);
    while(true){
        AtomicLockedLine expected_value = m_locked_lines.load();
        if (m_locked_lines.compare_exchange_weak(expected_value,{ (uint32_t)-1 , expected_value.m_process_line}))
            break;
    }
}

void NHTFlowCache::export_thread_function()noexcept{
    while(!m_exit){
        auto now = PacketClock::now();
        auto until = now + std::chrono::nanoseconds(m_export_sleep_time);
        std::this_thread::sleep_until(until);
        auto x = PacketClock::now() - now;
        for(auto i = 0u; !PacketClock::has_stopped() && i < x.count()/m_export_sleep_time; i++ )
            export_expired(PacketClock::now_as_timeval().tv_sec);
    }
}

void NHTFlowCache::export_expired_body(time_t ts) noexcept
{
    for (uint32_t i = m_timeout_idx; i < m_timeout_idx + m_line_size; i++) {
        if (!m_flow_table[i]->is_empty()
            && ts - m_flow_table[i]->m_flow.time_last.tv_sec >= m_inactive) {
            m_statistics.m_exported_on_periodic_scan++;
            prepare_and_export(
                i,
                has_tcp_eof_flags(m_flow_table[i]->m_flow)
                    ? ipxp::FlowEndReason::FLOW_END_EOF_DETECTED
                    : ipxp::FlowEndReason::FLOW_END_IDLE_TIMEOUT);
        }
    }
    m_timeout_idx = (m_timeout_idx + m_line_size) & (m_cache_size - 1);
}

/**
 * @brief Saves key values of flow.
 * @param pkt Incoming packet.
 * Saves key value and key length into attributes NHTFlowCache::key and NHTFlowCache::m_keylen
 */
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
    if (m_split_biflow)
        std::visit([&pkt](auto&& flow_key) { flow_key = pkt; }, m_key);
    else
        std::visit([&pkt](auto&& flow_key) { flow_key.save_sorted(pkt); }, m_key);
    return true;
}

void NHTFlowCache::print_report() const noexcept
{
    if (m_statistics.m_hits) {
        std::cout << "==================================================================\nTOTAL\n";
        std::cout << m_statistics;
    }
}
/**
 * @brief Statistics thread function.
 * @param stream Stream into which statistics will be written.
 * Prints statistics to stream in time interval defined by
 * NHTFlowCache::m_periodic_statistics_sleep_time. Must be called in separate thread
 */
void NHTFlowCache::export_periodic_statistics(std::ostream& stream) noexcept
{
    while (!m_exit) {
        std::this_thread::sleep_for(m_periodic_statistics_sleep_time);
        stream << m_statistics - m_last_statistics;
        m_last_statistics = m_statistics;
    }
}

CacheStatistics& NHTFlowCache::get_total_statistics() noexcept{
    return m_statistics;
}

CacheStatistics& NHTFlowCache::get_last_statistics() noexcept{
    return m_last_statistics;
}
} // namespace ipxp
