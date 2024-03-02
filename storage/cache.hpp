/**
 * \file cache.hpp
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
#ifndef IPXP_STORAGE_CACHE_HPP
#define IPXP_STORAGE_CACHE_HPP

#include <memory>
#include <optional>
#include <string>
#include <array>
#include <chrono>
#include <variant>
#include <thread>
#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/storage.hpp>
#include <ipfixprobe/utils.hpp>
#include "flowkeyv4.hpp"
#include "flowkeyv6.hpp"
#include "flowrecord.hpp"
#include "cachestatistics.hpp"
#include "cacheoptparser.hpp"
#include "flowendreason.hpp"
#include "cacheoptparser.hpp"
#include "cachestatistics.hpp"
#include "flowendreason.hpp"
#include "flowkeyv4.hpp"
#include "flowkeyv6.hpp"
#include "flowrecord.hpp"
#include <array>
#include <chrono>
#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/storage.hpp>
#include <ipfixprobe/utils.hpp>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <thread>
#include "fragmentationCache/fragmentationCache.hpp"
#include <sys/time.h>
#include "fstream"

namespace ipxp {

using namespace std::chrono_literals;
class NHTFlowCache : public StoragePlugin {
public:
    NHTFlowCache();
    ~NHTFlowCache() override;
    void init(const char* params) override;
    virtual void init(OptionsParser& parser);
    void set_queue(ipx_ring_t* queue) override;
    OptionsParser* get_parser() const;
    std::string get_name() const noexcept;
    int put_pkt(Packet& pkt) override;
    void export_expired(time_t ts) override;
    virtual void print_report() const noexcept;
    void finish() override;
    CacheStatistics& get_total_statistics() noexcept;
    CacheStatistics& get_last_statistics() noexcept;

protected:
    uint32_t m_cache_size; ///< Maximal count of records in cache
    uint32_t m_line_size; ///< Maximal count of records in one row
    uint32_t m_line_mask; ///< Line mask xored with flow index returns start of the row
    uint32_t m_insert_pos; ///< Insert position of new flow, if row has no empty space
    uint32_t m_qsize; ///< Export queue size
    uint32_t m_qidx; ///< Next position in export queue that will be exported
    uint32_t m_timeout_idx; ///< Index of the row where expired flow will be exported
    uint32_t m_active; ///< Active timeout
    uint32_t m_inactive; ///< Inactive timeout
    bool m_split_biflow; ///< If true, request and response packets between same ips will be counted
                         ///< belonging to different flows
    bool m_enable_fragmentation_cache; ///< If true, fragmentation cache will try to complete port
                                       ///< information for fragmented packet
    std::variant<FlowKeyV4, FlowKeyV6> m_key; ///< Key values of processed flow
    std::variant<FlowKeyV4, FlowKeyV6> m_key_inv; ///< Key values of processed flow with swapped
                                                  ///< source and destination addresses and ports
    std::vector<FlowRecord*>
        m_flow_table; ///< Pointers to flow records used for faster flow reorder operations
    std::vector<FlowRecord> m_flow_records; ///< Main memory of the cache
    CacheStatistics
        m_statistics; ///< Total statistics about cache efficiency from the program start
    CacheStatistics m_last_statistics; ///< Cache statistics for last
                                       ///< m_periodic_statistics_sleep_time amount of time
    CacheStatistics m_flood_statistics;
    bool m_exit; ///< Used for stopping background statistics thread
    std::chrono::duration<double>
        m_periodic_statistics_sleep_time; ///< Amount of time in which periodic statistics must
                                          ///< reset
    std::unique_ptr<std::thread> m_statistics_thread; ///< Pointer to periodic statistics thread
    FragmentationCache
        m_fragmentation_cache; ///< Fragmentation cache used for completing packets ports
    struct GraphExport{
        std::ofstream m_graph_datastream;
        std::ofstream m_graph_new_flows_datastream;
        std::ofstream m_graph_cusum_datastream;
        std::ofstream m_graph_cusum_threshold_datastream;
        timeval m_last_measurement{0,0};
        uint32_t m_interval = 1;
        CacheStatistics m_last_statistics;
    } m_graph_export;
    std::function<uint64_t(const void*,uint32_t)> m_hash_function;

    virtual void export_graph_data(const Packet& pkt);
    void try_to_fill_ports_to_fragmented_packet(Packet& packet);
    void allocate_tables();
    void export_periodic_statistics(std::ostream& stream) noexcept;
    void flush(Packet& pkt,uint32_t flow_index,int ret,bool source_flow,FlowEndReason reason) noexcept;
    virtual uint32_t free_place_in_full_line(uint32_t line_begin) noexcept;
    bool tcp_connection_reset(Packet& pkt, uint32_t flow_index, bool source) noexcept;
    void create_new_flow(uint32_t flow_index, Packet& pkt, uint64_t hashval) noexcept;
    bool update_flow(uint32_t flow_index, Packet& pkt,bool source) noexcept;
    virtual uint32_t make_place_for_record(uint32_t line_index) noexcept;
    std::tuple<bool, bool,uint32_t, uint64_t> find_flow_position(Packet& pkt) noexcept;
    virtual int insert_pkt(Packet& pkt) noexcept;
    bool timeouts_expired(Packet& pkt, uint32_t flow_index) noexcept;
    bool create_hash_key(const Packet& pkt) noexcept;
    void export_flow(uint32_t index);
    static uint8_t get_export_reason(Flow& flow);

    void cyclic_rotate_records(uint32_t begin,uint32_t end) noexcept;

    bool process_last_tcp_packet(Packet& pkt, uint32_t flow_index) noexcept;
    void get_opts_from_parser(const CacheOptParser& parser);
    std::pair<bool, uint32_t> find_existing_record(uint64_t hashval) const noexcept;
    virtual uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept;
    std::pair<bool, uint32_t> find_empty_place(uint32_t begin_line) const noexcept;
    void prepare_and_export(uint32_t flow_index, FlowEndReason reason) noexcept;
    uint64_t hash(const void* ptr, uint32_t len) const noexcept;
    void set_hash_function(std::function<uint64_t(const void*,uint32_t)> function) noexcept;

    static bool has_tcp_eof_flags(const Flow& flow) noexcept;
    static void test_attributes();
    virtual bool is_being_flooded(const Packet& Pkt) noexcept;


    struct FloodMeasurement{
        timeval m_last_measurement;
        uint64_t m_measurement_count = 0;
        uint64_t m_flows_created = 0;
        uint32_t m_last_mean = 0;
        uint64_t m_error_summ = 0;
        uint64_t m_error_summ2 = 0;
        uint64_t m_cusum = 0;
        const uint32_t m_interval_length = 5;
        const uint32_t m_span = 1000;
        const float m_coef = 2/(m_span/m_interval_length + 1);
        double m_deviation = 0;
        const uint32_t m_threshold = 5;
        const double m_min = 7000;
    } m_flood_measurement;
};

} // namespace ipxp
#endif /* IPXP_STORAGE_CACHE_HPP */
