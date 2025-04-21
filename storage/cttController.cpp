/**
* \file
 * \author Damir Zainullin <zaidamilda@gmail.com>
 * \brief CttController implementation.
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

#include "cttController.hpp"
#include <cstring>
#include <endian.h>
#include <iostream>
#ifdef WITH_CTT

namespace ipxp {

CttController::CttController(const std::string& nfb_dev, unsigned ctt_comp_index) {
    //m_commander = std::make_unique<ctt::AsyncCommander>(ctt::NfbDebugParams{"/tmp/ctt.out", nfb_dev, ctt_comp_index});
    //cpu_set_t cpuset;      
    //CPU_ZERO(&cpuset);     
    //CPU_SET(14, &cpuset);   
    m_commander = std::make_unique<ctt::AsyncCommander>(ctt::NfbParamsFast{nfb_dev, ctt_comp_index});
    try {
        // Get UserInfo to determine key, state, and state_mask sizes
        ctt::UserInfo user_info = m_commander->get_user_info();
        m_key_size_bytes = (user_info.key_bit_width + 7) / 8;
        m_state_size_bytes = (user_info.state_bit_width + 7) / 8;
        if (m_state_size_bytes != sizeof(feta::CttRecord)) {
            throw std::runtime_error("Size of CTT state does not match the expected size.");
        }
        m_state_mask_size_bytes = (user_info.state_mask_bit_width + 7) / 8;

        // Enable the CTT
        std::future<void> enable_future = m_commander->enable(true);
        enable_future.wait();
    }
    catch (const std::exception& e) {
        throw;
    }
}

void CttController::create_record(const Flow& flow, uint8_t dma_channel, feta::OffloadMode offload_mode)
{
    try {
        std::vector<std::byte> key = assemble_key(flow.flow_hash_ctt);
        std::vector<std::byte> state = assemble_state(
              offload_mode,
              feta::MetaType::FULL_META,
              flow, dma_channel);
        m_commander->write_record(std::move(key), std::move(state));
        MAYBE_DISABLED_CODE(std::cout << "Create record with key" << std::hex << flow.flow_hash_ctt << std::endl;)
    }
    catch (const std::exception& e) {
        throw;
    }
}

void CttController::get_state(uint64_t flow_hash_ctt)
{
    return export_record(flow_hash_ctt);
    MAYBE_DISABLED_CODE(std::cout << "Getting state of " << std::hex << flow_hash_ctt << std::endl;)
    try {
        std::vector<std::byte> key = assemble_key(flow_hash_ctt);
        m_commander->export_record(std::move(key));
    }
    catch (const std::exception& e) {
        throw;
    }
}

void CttController::remove_record_without_notification(uint64_t flow_hash_ctt)
{
    try {
        std::vector<std::byte> key = assemble_key(flow_hash_ctt);
        m_commander->delete_record(std::move(key));
        MAYBE_DISABLED_CODE(std::cout << "Deliting without export key " << std::hex << flow_hash_ctt << std::endl;)
    }
    catch (const std::exception& e) {
        throw;
    }
}

void CttController::export_record(uint64_t flow_hash_ctt)
{
    try {
        std::vector<std::byte> key = assemble_key(flow_hash_ctt);
        m_commander->export_and_delete_record(std::move(key));
        MAYBE_DISABLED_CODE(std::cout << "Exporting and deliting key " << std::hex << flow_hash_ctt << std::endl;)

    }
    catch (const std::exception& e) {
        throw;
    }
}

std::pair<std::vector<std::byte>, std::vector<std::byte>>
CttController::get_key_and_state(uint64_t flow_hash_ctt, const Flow& flow, uint8_t dma_channel)
{
    return {assemble_key(flow_hash_ctt), assemble_state(
          feta::OffloadMode::TRIM_PACKET_META,
          feta::MetaType::FULL_META,
          flow, dma_channel)};
}

std::vector<std::byte> CttController::assemble_key(uint64_t flow_hash_ctt)
{
    return std::vector<std::byte>(reinterpret_cast<const std::byte*>(&flow_hash_ctt),
        reinterpret_cast<const std::byte*>(&flow_hash_ctt) + m_key_size_bytes);
    std::vector<std::byte> key(m_key_size_bytes, std::byte(0));
    for (size_t i = 0; i < sizeof(flow_hash_ctt) && i < m_key_size_bytes; ++i) {
        key[i] = static_cast<std::byte>((flow_hash_ctt >> (8 * i)) & 0xFF);
    }
    return key;
}

std::vector<std::byte> CttController::assemble_state(
    feta::OffloadMode offload_mode, feta::MetaType meta_type, const Flow& flow, uint8_t dma_channel)
{
    std::vector<std::byte> state(sizeof(feta::CttRecord), std::byte(0));
    feta::CttRecord record;
    record.ts_first.time_sec = flow.time_first.tv_sec;
    record.ts_first.time_ns = flow.time_first.tv_usec * 1000;
    record.ts_last.time_sec = flow.time_last.tv_sec;
    record.ts_last.time_ns = flow.time_last.tv_usec * 1000;
    const size_t ip_length = flow.ip_version == IP::v4 ? 4 : 16;
    ///TODO FIX IP ADDRESSES
    /*if (flow.ip_version == IP::v4) {
        record.ip_src = {};
        record.ip_dst = {};
        std::memcpy(&record.ip_src[3], &flow.src_ip, ip_length);
        std::memcpy(&record.ip_dst[3], &flow.dst_ip, ip_length);
    } else {
        std::memcpy(record.ip_src.data(), &flow.src_ip, ip_length);
        std::memcpy(record.ip_dst.data(), &flow.dst_ip, ip_length);
    }*/
    std::memset(record.ip_src.data(), 0, 16);
    std::memset(record.ip_dst.data(), 0, 16);
    std::memcpy(record.ip_src.data(), &flow.src_ip, ip_length);
    std::memcpy(record.ip_dst.data(), &flow.dst_ip, ip_length);
    record.port_src = flow.src_port;
    record.port_dst = flow.dst_port;
    record.vlan_tci = flow.vlan_id;
    record.l4_proto = flow.ip_proto;
    record.ip_ver = flow.ip_version == IP::v4 ? feta::IpVersion::IPV4 : feta::IpVersion::IPV6;
    record.vlan_vld = flow.vlan_id ? 1 : 0;
    record.offload_mode = offload_mode;
    record.meta_type = meta_type;
    record.limit_size = 0;
    record.dma_chan = dma_channel;
    //record.bytes = flow.src_bytes;
    //record.bytes_rev = flow.dst_bytes;
    //record.pkts = flow.src_packets;
    //record.pkts_rev = flow.dst_packets;
    //record.tcp_flags = flow.src_tcp_flags;
    //record.tcp_flags_rev = flow.dst_tcp_flags;
    
    record.bytes = 0;
    record.bytes_rev = 0;
    record.pkts = 0;
    record.pkts_rev = 0;
    record.tcp_flags = 0;
    record.tcp_flags_rev = 0;
    feta::CttRecord::serialize(record, state.data());
    /*CttState* ctt_state = reinterpret_cast<CttState*>(state.data());
    const size_t ip_length = flow.ip_version == IP::v4 ? 4 : 16;

    ctt_state->dma_channel = dma_channel;
    ctt_state->time_first.tv_sec = htole32(static_cast<uint32_t>(flow.time_first.tv_sec));
    ctt_state->time_first.tv_usec = htole32(static_cast<uint32_t>(flow.time_first.tv_usec));
    ctt_state->time_last.tv_sec = htole32(static_cast<uint32_t>(flow.time_last.tv_sec));
    ctt_state->time_last.tv_usec = htole32(static_cast<uint32_t>(flow.time_last.tv_usec));
    std::reverse_copy(reinterpret_cast<const uint8_t*>(&flow.src_ip),
        reinterpret_cast<const uint8_t*>(&flow.src_ip) + ip_length, reinterpret_cast<uint8_t*>(&ctt_state->src_ip));
    std::reverse_copy(reinterpret_cast<const uint8_t*>(&flow.dst_ip),
        reinterpret_cast<const uint8_t*>(&flow.dst_ip) + ip_length, reinterpret_cast<uint8_t*>(&ctt_state->dst_ip));
    ctt_state->ip_version = flow.ip_version == IP::v4 ? 0 : 1;
    ctt_state->ip_proto = flow.ip_proto;
    ctt_state->src_port = htole16(flow.src_port);
    ctt_state->dst_port = htole16(flow.dst_port);
    ctt_state->tcp_flags = flow.src_tcp_flags;
    ctt_state->tcp_flags_rev = flow.dst_tcp_flags;
    ctt_state->packets = htole16(flow.src_packets);
    ctt_state->packets_rev = htole16(flow.dst_packets);
    ctt_state->bytes = htole32(flow.src_bytes);
    ctt_state->bytes_rev = htole32(flow.dst_bytes);
    ctt_state->limit_size = htole16(0);
    ctt_state->offload_mode = offload_mode;
    ctt_state->meta_type = meta_type;
    ctt_state->was_exported = 0;*/
    return state;
}


CttController::~CttController() noexcept
{
    /*if (!m_commander) {
        return;
    }
    std::future<void> enable_future = m_commander->enable(false);
    enable_future.wait();
    m_commander.reset();*/
}

} // ipxp

#endif /* WITH_CTT */
