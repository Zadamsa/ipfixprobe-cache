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
#ifndef IPXP_STORAGE_CACHE_HPP
#define IPXP_STORAGE_CACHE_HPP

#include <memory>
#include <optional>
#include <string>
#include <array>
#include <chrono>

#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/options.hpp>
#include <ipfixprobe/storage.hpp>
#include <ipfixprobe/utils.hpp>
#include "flowkeyv4.hpp"
#include "flowkeyv6.hpp"
#include "flowrecord.hpp"
#include "cachestatistics.hpp"
#include "cacheoptparser.hpp"
#include "flowendreason.hpp"

namespace ipxp {

#ifdef IPXP_FLOW_CACHE_SIZE
static const uint32_t DEFAULT_FLOW_CACHE_SIZE = IPXP_FLOW_CACHE_SIZE;
#else
static const uint32_t DEFAULT_FLOW_CACHE_SIZE = 17; // 131072 records total
#endif /* IPXP_FLOW_CACHE_SIZE */

#ifdef IPXP_FLOW_LINE_SIZE
static const uint32_t DEFAULT_FLOW_LINE_SIZE = IPXP_FLOW_LINE_SIZE;
#else
static const uint32_t DEFAULT_FLOW_LINE_SIZE = 4; // 16 records per line
#endif /* IPXP_FLOW_LINE_SIZE */

using namespace std::chrono_literals;
class NHTFlowCache : public StoragePlugin {
public:
    NHTFlowCache();
    ~NHTFlowCache() override;
    void init(const char* params) override;
    void close() override;
    void set_queue(ipx_ring_t* queue) override;
    OptionsParser* get_parser() const;
    std::string get_name() const noexcept;
    int put_pkt(Packet& pkt) override;
    void export_expired(time_t ts) override;
    void print_report() const noexcept;
private:
    uint32_t m_cache_size = 0;
    uint32_t m_line_size = 0;
    uint32_t m_line_mask = 0;
    uint32_t m_line_new_idx = 0;
    uint32_t m_qsize = 0;
    uint32_t m_qidx = 0;
    uint32_t m_timeout_idx = 0;
    uint32_t m_active = 0;
    uint32_t m_inactive = 0;
    bool m_split_biflow = false;
    uint8_t m_keylen = 0;
    uint8_t m_key[max<size_t>(sizeof(FlowKeyV4), sizeof(FlowKeyV6))];
    uint8_t m_key_inv[max<size_t>(sizeof(FlowKeyV4), sizeof(FlowKeyV6))];
    std::unique_ptr<FlowRecord*[]> m_flow_table = nullptr;
    std::unique_ptr<FlowRecord[]> m_flow_records = nullptr;
    CacheStatistics m_statistics = {},m_last_statistics = {};
    bool* m_exit = new bool{false};
    const std::chrono::duration<double> m_periodic_statistics_sleep_time = 1s;

    void allocate_tables();
    void export_periodic_statistics(std::ostream& stream) const noexcept;
    void flush(Packet& pkt,size_t flow_index,int ret,bool source_flow,FlowEndReason reason) noexcept;
    uint32_t shift_records(uint32_t flow_index,uint32_t line_begin,uint32_t line_end) noexcept;
    bool tcp_connection_reset(Packet& pkt,uint32_t flow_index) noexcept;
    void create_new_flow(uint32_t flow_index,Packet& pkt,uint64_t hashval) noexcept;
    bool update_flow(uint32_t flow_index,Packet& pkt) noexcept;
    uint32_t make_place_for_record(uint32_t line_index,uint32_t  next_line) noexcept;
    std::tuple<bool,size_t,size_t,size_t,size_t> find_flow_position(Packet& pkt) noexcept;
    int insert_pkt(Packet& pkt) noexcept;
    bool timeouts_expired(Packet& pkt,size_t flow_index) noexcept;
    static bool has_tcp_eof_flags(const Flow& flow) noexcept;

    bool create_hash_key(const Packet& pkt) noexcept;
    void export_flow(size_t index);
    static uint8_t get_export_reason(Flow& flow);
    void finish() override;
    void get_opts_from_parser(const CacheOptParser& parser);

    std::pair<bool, uint32_t>
    find_existing_record(uint32_t begin_line, uint32_t end_line, uint64_t hashval) const noexcept;
    virtual uint32_t
    enhance_existing_flow_record(uint32_t flow_index, uint32_t line_index) noexcept;
    std::pair<bool, uint32_t>
    find_empty_place(uint32_t begin_line, uint32_t end_line) const noexcept;
    virtual uint32_t put_into_free_place(
        uint32_t flow_index,
        bool empty_place_found,
        uint32_t begin_line,
        uint32_t end_line) noexcept;

    bool process_last_tcp_packet(Packet& pkt, uint32_t flow_index) noexcept;
    //virtual bool create_new_flow(uint32_t flow_index, Packet& pkt, uint64_t hashval) noexcept;
    virtual bool flush_and_update_flow(uint32_t flow_index, Packet& pkt) noexcept;
    virtual void prepare_and_export(uint32_t flow_index) noexcept;
    virtual void prepare_and_export(uint32_t flow_index, uint32_t reason) noexcept;

    static void test_attributes();
};


} // namespace ipxp
#endif /* IPXP_STORAGE_CACHE_HPP */
