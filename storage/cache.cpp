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

#include <ipfixprobe/ring.h>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <ratio>
#include <sys/time.h>
#include <optional>
#include <endian.h>
#include <algorithm>

#include "xxhash.h"
#include "fragmentationCache/timevalUtils.hpp"
#include "cacheRowSpan.hpp"
#include "flowKeyFactory.tpp"

namespace ipxp {

__attribute__((constructor)) static void register_this_plugin()
{
   static PluginRecord rec = PluginRecord("cache", [](){return new NHTFlowCache();});
   register_plugin(&rec);
}

OptionsParser * NHTFlowCache::get_parser() const
{
    return new CacheOptParser();
}

std::string NHTFlowCache::get_name() const noexcept
{
    return "cache";
}

NHTFlowCache::NHTFlowCache()
{
}

NHTFlowCache::~NHTFlowCache()
{
   NHTFlowCache::close();
   print_report();
}

void NHTFlowCache::get_parser_options(CacheOptParser& parser) noexcept
{
    m_cache_size = parser.m_cache_size;
    m_line_size = parser.m_line_size;
    m_active = parser.m_active;
    m_inactive = parser.m_inactive;
    m_line_mask = (m_cache_size - 1) & ~(m_line_size - 1);
    m_new_flow_insert_index = m_line_size / 2;
    m_split_biflow = parser.m_split_biflow;
    m_enable_fragmentation_cache = parser.m_enable_fragmentation_cache;
}

void NHTFlowCache::allocate_table()
{
    try {
        m_flow_table.resize(m_cache_size + m_queue_size);
        m_flows.resize(m_cache_size + m_queue_size);
        std::for_each(m_flow_table.begin(), m_flow_table.end(), [index = 0, this](FlowRecord*& flow) mutable  {
            flow = &m_flows[index++];
        });
    } catch (std::bad_alloc &e) {
        throw PluginError("not enough memory for flow cache allocation");
    }
}

void NHTFlowCache::init(const char *params)
{
   CacheOptParser parser;
   try {
      parser.parse(params);
   } catch (ParserError &e) {
      throw PluginError(e.what());
   }

   get_parser_options(parser);
   if (m_export_queue == nullptr) {
      throw PluginError("output queue must be set before init");
   }
   if (m_line_size > m_cache_size) {
      throw PluginError("flow cache line size must be greater or equal to cache size");
   }
   if (m_cache_size == 0) {
      throw PluginError("flow cache won't properly work with 0 records");
   }
   allocate_table();

   if (m_enable_fragmentation_cache) {
      try {
         m_fragmentation_cache = FragmentationCache(parser.m_frag_cache_size, parser.m_frag_cache_timeout);
      } catch (std::bad_alloc &e) {
         throw PluginError("not enough memory for fragment cache allocation");
      }
   }
}

void NHTFlowCache::close()
{
   m_flows.clear();
   m_flow_table.clear();
}

void NHTFlowCache::set_queue(ipx_ring_t *queue)
{
   m_export_queue = queue;
   m_queue_size = ipx_ring_size(queue);
}

void NHTFlowCache::export_flow(size_t flow_index)
{
   export_flow(flow_index, get_export_reason(m_flow_table[flow_index]->m_flow));
}

void NHTFlowCache::export_flow(size_t flow_index, int reason)
{
   MAYBE_DISABLED_CODE(std::cout << "Exporting ipfixprobe flow" << std::endl;)
#ifdef WITH_CTT
   m_ctt_stats.real_processed_packets += m_flow_table[flow_index]->m_flow.src_packets + m_flow_table[flow_index]->m_flow.dst_packets;
#endif /* WITH_CTT */
   m_flow_table[flow_index]->m_flow.end_reason = reason;
   update_flow_record_stats(m_flow_table[flow_index]->m_flow.src_packets + m_flow_table[flow_index]->m_flow.dst_packets);
   update_flow_end_reason_stats(m_flow_table[flow_index]->m_flow.end_reason);
   m_cache_stats.exported++;
   push_to_export_queue(flow_index);
   m_flow_table[flow_index]->erase();
   m_cache_stats.flows_in_cache--;
   m_cache_stats.total_exported++;
}

void NHTFlowCache::push_to_export_queue(size_t flow_index) noexcept
{
   ipx_ring_push(m_export_queue, &m_flow_table[flow_index]->m_flow);
   std::swap(m_flow_table[flow_index], m_flow_table[m_cache_size + m_queue_index]);
   m_queue_index = (m_queue_index + 1) % m_queue_size;
}

void NHTFlowCache::finish()
{
   MAYBE_DISABLED_CODE(std::cout << "\n\nFinishing cache";)
   std::for_each(m_flow_table.begin(), m_flow_table.begin() + m_cache_size, [&](FlowRecord*& flow_record) {
      if (!flow_record->is_empty()) {
#ifdef WITH_CTT
         if (flow_record->is_in_ctt) {
            m_ctt_stats.total_requests_count++;
            m_ctt_controller->remove_record_without_notification(flow_record->m_flow.flow_hash_ctt);
         }
#endif /* WITH_CTT */
         plugins_pre_export(flow_record->m_flow);
         export_flow(&flow_record - m_flow_table.data(), FLOW_END_FORCED);
      }
   });
}

#ifdef WITH_CTT
void NHTFlowCache::prefinish_signal() noexcept
{
   constexpr size_t BLOCK_SIZE = 64;
   for(size_t current_index = m_prefinish_index; current_index < m_prefinish_index + BLOCK_SIZE; current_index++) {
      FlowRecord* flow_record = m_flow_table[current_index];
      if (!flow_record->is_empty() && flow_record->is_in_ctt && flow_record->offload_mode == feta::OffloadMode::DROP_PACKET_DROP_META) {
         m_ctt_stats.total_requests_count++;
         m_ctt_controller->get_state(flow_record->m_flow.flow_hash_ctt);
      }  
   };
   m_prefinish_index = (m_prefinish_index + BLOCK_SIZE) % m_cache_size;
}
#endif /* WITH_CTT */

void NHTFlowCache::export_and_reuse_flow(size_t flow_index) noexcept
{
   push_to_export_queue(flow_index);
   m_flow_table[flow_index]->m_flow.remove_extensions();
   *m_flow_table[flow_index] = *m_flow_table[m_cache_size + m_queue_index];
   m_flow_table[flow_index]->m_flow.m_exts = nullptr;
   m_flow_table[flow_index]->reuse(); // Clean counters, set time first to last
}

void NHTFlowCache::flush(Packet &pkt, size_t flow_index, int return_flags)
{
   m_cache_stats.flushed++;

   if (return_flags == ProcessPlugin::FlowAction::FLUSH_WITH_REINSERT) {
#ifdef WITH_CTT
      /*if (m_flow_table[flow_index]->is_in_ctt && !m_flow_table[flow_index]->is_waiting_for_export) {
         m_flow_table[flow_index]->is_waiting_for_export = true;
         m_ctt_controller->remove_record_without_notification(m_flow_table[flow_index]->m_flow.flow_hash_ctt);
      }*/
#endif /* WITH_CTT */
      export_and_reuse_flow(flow_index);
      m_flow_table[flow_index]->update(pkt); // Set new counters from packet
      const size_t post_create_return_flags = plugins_post_create(m_flow_table[flow_index]->m_flow, pkt);
      if (post_create_return_flags & ProcessPlugin::FlowAction::FLUSH) {
         flush(pkt, flow_index, post_create_return_flags);
      }
      return;
   }
   try_to_export(flow_index, false, pkt.ts, FLOW_END_FORCED);
}

NHTFlowCache::FlowSearch
NHTFlowCache::find_row(const FlowKey& key) noexcept
{
   const size_t hash_value = key.hash();
   const size_t first_flow_in_row = hash_value & m_line_mask;
   const CacheRowSpan row(&m_flow_table[first_flow_in_row], m_line_size);
   if (const std::optional<size_t> flow_index = row.find_by_hash(hash_value); flow_index.has_value()) {
      return {row, first_flow_in_row + *flow_index, hash_value};
   }
   return {row, std::nullopt, hash_value};
}

std::pair<NHTFlowCache::FlowSearch, bool>
NHTFlowCache::find_flow_index(const FlowKey& key, const FlowKey& key_reversed) noexcept
{
   const FlowSearch direct_search = find_row(key);
   if (direct_search.flow_index.has_value() || m_split_biflow) {
      return {direct_search, true};
   }

   const FlowSearch reverse_search = find_row(key_reversed);
   if (reverse_search.flow_index.has_value()) {
      return {reverse_search, false};
   }

   return {direct_search, true};
}

std::pair<NHTFlowCache::FlowSearch, bool>
NHTFlowCache::find_flow_index(const FlowKey& sorted_key, const bool swapped) noexcept
{
   const FlowSearch search = find_row(sorted_key);
   if (search.flow_index.has_value()) {
      return {search, m_flow_table[*search.flow_index]->m_flow.swapped == swapped};
   }
   return {search, swapped};
}

static bool is_tcp_connection_restart(const Packet& packet, const Flow& flow) noexcept
{
   constexpr uint8_t TCP_FIN = 0x01;
   constexpr uint8_t TCP_RST = 0x04;
   constexpr uint8_t TCP_SYN = 0x02;
   const uint8_t flags = packet.source_pkt ? flow.src_tcp_flags : flow.dst_tcp_flags;
   return (packet.tcp_flags & TCP_SYN) && (flags & (TCP_FIN | TCP_RST));
}

bool NHTFlowCache::try_to_export_on_inactive_timeout(size_t flow_index, const timeval& now) noexcept
{
   if (now.tv_sec - m_flow_table[flow_index]->m_flow.time_last.tv_sec < 0) {
      MAYBE_DISABLED_CODE(std::cout << "Negative timestamp " <<  now.tv_sec - m_flow_table[flow_index]->m_flow.time_last.tv_sec << std::endl;)
   }
   if (!m_flow_table[flow_index]->is_empty() && now.tv_sec - m_flow_table[flow_index]->m_flow.time_last.tv_sec >= m_inactive) {
#ifdef WITH_CTT
      if (m_flow_table[flow_index]->is_in_ctt && m_flow_table[flow_index]->is_waiting_ctt_response) {
         if (now > m_flow_table[flow_index]->last_request_time + CTT_REQUEST_TIMEOUT) {
            m_ctt_stats.total_requests_count++;
            m_ctt_stats.lost_requests_count++;
            m_ctt_controller->get_state(m_flow_table[flow_index]->m_flow.flow_hash_ctt);
            m_flow_table[flow_index]->last_request_time = now;
         }
         return false;
      }
      if (m_flow_table[flow_index]->is_in_ctt && m_flow_table[flow_index]->offload_mode == feta::OffloadMode::DROP_PACKET_DROP_META 
            && (m_flow_table[flow_index]->m_flow.time_last > m_flow_table[flow_index]->last_request_time
            || m_flow_table[flow_index]->last_request_time.tv_sec - m_flow_table[flow_index]->m_flow.time_last.tv_sec < m_inactive)) {
         m_ctt_stats.total_requests_count++;
         m_ctt_controller->get_state(m_flow_table[flow_index]->m_flow.flow_hash_ctt);
         m_flow_table[flow_index]->last_request_time = now;
         m_flow_table[flow_index]->is_waiting_ctt_response = true;
         return false;
         
      }
#endif /* WITH_CTT */   
      return try_to_export(flow_index, false, now);
   }
   return false;
}

#ifdef WITH_CTT
std::optional<feta::OffloadMode> NHTFlowCache::get_offload_mode(size_t flow_index) noexcept
{
   //return feta::OffloadMode::TRIM_PACKET_META;
   return std::nullopt;
   /*static int count = 0;
   if (count++ % 100 != 0) {
      m_ctt_stats.drop_packet_offloaded++;
      return feta::OffloadMode::DROP_PACKET_DROP_META;
   } else {
      m_ctt_stats.trim_packet_offloaded++;
      return feta::OffloadMode::TRIM_PACKET_META;
   }*/
   //return feta::OffloadMode::TRIM_PACKET_META;
   //return feta::OffloadMode::DROP_PACKET_DROP_META;
   //return std::nullopt;
   if (!m_flow_table[flow_index]->can_be_offloaded) {
      return std::nullopt;
   }
   /*if (m_flow_table[flow_index]->m_flow.src_packets + m_flow_table[flow_index]->m_flow.dst_packets > 1000) {
      m_ctt_stats.drop_packet_offloaded++;
      return feta::OffloadMode::DROP_PACKET_DROP_META ;
   }*/
   if (m_flow_table[flow_index]->m_flow.src_packets + m_flow_table[flow_index]->m_flow.dst_packets > 1000) {
      m_ctt_stats.trim_packet_offloaded++;
      return feta::OffloadMode::TRIM_PACKET_META;
   }
   
   return std::nullopt;
}
#endif /*WITH_CTT*/

void NHTFlowCache::create_record(const Packet& packet, size_t flow_index, size_t hash_value) noexcept
{
   m_cache_stats.flows_in_cache++;
   m_flow_table[flow_index]->create(packet, hash_value);
   const size_t post_create_return_flags = plugins_post_create(m_flow_table[flow_index]->m_flow, packet);
   if (post_create_return_flags & ProcessPlugin::FlowAction::FLUSH) {
      export_flow(flow_index);
      m_cache_stats.flushed++;
      return;
   }
#ifdef WITH_CTT
   // if metadata are valid, add flow hash ctt to the flow record
   if (!packet.cttmeta_valid) {
      return;
   }
   m_flow_table[flow_index]->m_flow.flow_hash_ctt = packet.cttmeta.flow_hash;
   if (const std::optional<feta::OffloadMode> offload_mode = get_offload_mode(flow_index); offload_mode.has_value()) {
      offload_flow_to_ctt(flow_index, *offload_mode);
   }
#endif /* WITH_CTT */
}

#ifdef WITH_CTT
void NHTFlowCache::offload_flow_to_ctt(size_t flow_index, feta::OffloadMode offload_mode) noexcept 
{
   m_ctt_stats.total_requests_count++;
   m_ctt_controller->create_record(m_flow_table[flow_index]->m_flow, m_dma_channel, offload_mode);
   m_ctt_stats.flows_offloaded++;
   m_flow_table[flow_index]->is_in_ctt = true;
   m_flow_table[flow_index]->offload_mode = offload_mode;
}

void NHTFlowCache::try_to_add_flow_to_ctt(size_t flow_index) noexcept
{
   if (m_flow_table[flow_index]->is_in_ctt || m_flow_table[flow_index]->m_flow.flow_hash_ctt == 0) {
      return;
   }
   if (const std::optional<feta::OffloadMode> offload_mode = get_offload_mode(flow_index); offload_mode.has_value()) {
      offload_flow_to_ctt(flow_index, *offload_mode);
   }
}
#endif /* WITH_CTT */

int NHTFlowCache::update_flow(Packet& packet, size_t flow_index, bool flow_is_waiting_for_export) noexcept
{
   if (!flow_is_waiting_for_export && is_tcp_connection_restart(packet, m_flow_table[flow_index]->m_flow)) {
      if (try_to_export(flow_index, false, packet.ts, FLOW_END_EOF)) {
         put_pkt(packet);
         return 0;
      }
   }

   /* Check if flow record is expired (inactive timeout). */
   if (!flow_is_waiting_for_export
         && try_to_export_on_inactive_timeout(flow_index, packet.ts)) {
      return put_pkt(packet);
   }

   if (!flow_is_waiting_for_export
         && try_to_export_on_active_timeout(flow_index, packet.ts)) {
      return put_pkt(packet);
   }

   const size_t pre_update_return_flags = plugins_pre_update(m_flow_table[flow_index]->m_flow, packet);
   if ((pre_update_return_flags & ProcessPlugin::FlowAction::FLUSH)
      && !flow_is_waiting_for_export) {
      flush(packet, flow_index, pre_update_return_flags);
      return 0;
   }

   m_flow_table[flow_index]->update(packet);
#ifdef WITH_CTT
   try_to_add_flow_to_ctt(flow_index);     
#endif /* WITH_CTT */
   const size_t post_update_return_flags = plugins_post_update(m_flow_table[flow_index]->m_flow, packet);
   if ((post_update_return_flags & ProcessPlugin::FlowAction::FLUSH)
         && !flow_is_waiting_for_export) {
      flush(packet, flow_index, post_update_return_flags);
      return 0;
   }

   export_expired(packet.ts);
   return 0;
}
#ifdef WITH_CTT
bool NHTFlowCache::try_to_export_delayed_flow(const Packet& packet, size_t flow_index) noexcept
{
   if (!m_flow_table[flow_index]->is_in_ctt) {
      return false;
   }
   if (m_flow_table[flow_index]->is_waiting_ctt_response && packet.ts > m_flow_table[flow_index]->last_request_time + CTT_REQUEST_TIMEOUT) {
      plugins_pre_export(m_flow_table[flow_index]->m_flow);
      export_flow(flow_index);
      return true;
   }
   return false;
}
#endif /* WITH_CTT */

bool NHTFlowCache::try_to_export(size_t flow_index, bool call_pre_export, const timeval& now) noexcept
{
   return try_to_export(flow_index, call_pre_export, now, get_export_reason(m_flow_table[flow_index]->m_flow));
}

#ifdef WITH_CTT
void NHTFlowCache::send_export_request_to_ctt(size_t ctt_flow_hash) noexcept
{
   m_ctt_stats.total_requests_count++;
   m_ctt_controller->export_record(ctt_flow_hash);
}
#endif /* WITH_CTT */

bool NHTFlowCache::try_to_export(size_t flow_index, bool call_pre_export, const timeval& now, int reason) noexcept
{
#ifdef WITH_CTT
   if (m_flow_table[flow_index]->is_in_ctt) {
      if (!m_flow_table[flow_index]->is_waiting_ctt_response) {
         m_flow_table[flow_index]->is_waiting_ctt_response = true;
         m_ctt_stats.total_requests_count++;
         m_ctt_controller->export_record(m_flow_table[flow_index]->m_flow.flow_hash_ctt);
         m_flow_table[flow_index]->last_request_time = now; //{now.tv_sec + 1, now.tv_usec};
         return false;
      }
      if (m_flow_table[flow_index]->last_request_time + CTT_REQUEST_TIMEOUT > now) {
         return false;
      }
      m_flow_table[flow_index]->is_waiting_ctt_response = false;
   }
#endif /* WITH_CTT */
   if (call_pre_export) {
      plugins_pre_export(m_flow_table[flow_index]->m_flow);
   }
   export_flow(flow_index, reason);
   return true;
}

#ifdef WITH_CTT

int convert_ctt_export_reason_to_ipfxiprobe(feta::ExportReason ctt_reason, feta::MuExportReason mu_reason) noexcept
{
   switch (ctt_reason) {
      case feta::ExportReason::EXPORT_BY_SW:
         return FLOW_END_FORCED;
      case feta::ExportReason::FULL_CTT:
         return FLOW_END_FORCED;
      case feta::ExportReason::EXPORT_BY_MU:
         if (static_cast<uint8_t>(mu_reason) & static_cast<uint8_t>(feta::MuExportReason::COUNTER_OVERFLOW)) {
            return FLOW_END_FORCED;
         }
         if (static_cast<uint8_t>(mu_reason) & static_cast<uint8_t>(feta::MuExportReason::TCP_CONN_END)) {
            return FLOW_END_EOF;
         }
         if (static_cast<uint8_t>(mu_reason) & static_cast<uint8_t>(feta::MuExportReason::ACTIVE_TIMEOUT)) {
               return FLOW_END_ACTIVE;
         }
      default:
         return FLOW_END_NO_RES;
   }
}

void NHTFlowCache::update_ctt_export_stats(feta::ExportReason ctt_reason, feta::MuExportReason mu_reason) noexcept
{
   switch (ctt_reason) {
      case feta::ExportReason::EXPORT_BY_SW:
         m_ctt_stats.export_reasons.by_request++;
         break;
      case feta::ExportReason::FULL_CTT:
         m_ctt_stats.export_reasons.ctt_full++;
         break;
      case feta::ExportReason::RESERVED:
         m_ctt_stats.export_reasons.reserved++;
         break;
      case feta::ExportReason::EXPORT_BY_MU:
         if (static_cast<uint8_t>(mu_reason) & static_cast<uint8_t>(feta::MuExportReason::COUNTER_OVERFLOW)) {
            m_ctt_stats.export_reasons.counter_overflow++;
         }
         if (static_cast<uint8_t>(mu_reason) & static_cast<uint8_t>(feta::MuExportReason::TCP_CONN_END)) {
            m_ctt_stats.export_reasons.tcp_eof++;
         }
         if (static_cast<uint8_t>(mu_reason) & static_cast<uint8_t>(feta::MuExportReason::ACTIVE_TIMEOUT)) {
            m_ctt_stats.export_reasons.active_timeout++;
         }
         break;
   }
}

static bool is_counter_overflow(feta::ExportReason ctt_reason, feta::MuExportReason mu_reason) noexcept
{
   return ctt_reason == feta::ExportReason::EXPORT_BY_MU && (static_cast<uint8_t>(mu_reason) & static_cast<uint8_t>(feta::MuExportReason::COUNTER_OVERFLOW));
}

static void update_packet_counters_from_active_timeout_external_export(Flow& flow, const feta::CttRecord& state) noexcept
{
   flow.src_packets += state.pkts;
   flow.dst_packets += state.pkts_rev;
   flow.src_bytes += state.bytes;
   flow.dst_bytes += state.bytes_rev;
   flow.time_last.tv_sec = state.ts_last.time_sec;
   flow.time_last.tv_usec = state.ts_last.time_ns / 1000;
   if (flow.ip_proto == 6) {
      flow.src_tcp_flags |= state.tcp_flags;
      flow.dst_tcp_flags |= state.tcp_flags_rev;
   }
}

static void update_packet_counters_from_external_export(Flow& flow, const feta::CttRecord& state) noexcept
{
   flow.src_packets = state.pkts;
   flow.dst_packets = state.pkts_rev;
   flow.src_bytes = state.bytes;
   flow.dst_bytes = state.bytes_rev;
   flow.time_last.tv_sec = state.ts_last.time_sec;
   flow.time_last.tv_usec = state.ts_last.time_ns / 1000;
   if (flow.ip_proto == 6) {
      flow.src_tcp_flags = state.tcp_flags;
      flow.dst_tcp_flags = state.tcp_flags_rev;
   }
}

static bool is_hash_collision(feta::ExportReason ctt_reason, feta::MuExportReason mu_reason) noexcept
{
   return ctt_reason == feta::ExportReason::EXPORT_BY_MU && mu_reason == feta::MuExportReason::FLOW_COLLISION;
}

static bool is_tcp_restart(feta::ExportReason ctt_reason, feta::MuExportReason mu_reason) noexcept
{
   return ctt_reason == feta::ExportReason::EXPORT_BY_MU && mu_reason == feta::MuExportReason::TCP_CONN_END;
}

static bool is_active_timeout(feta::ExportReason ctt_reason, feta::MuExportReason mu_reason) noexcept
{
   return ctt_reason == feta::ExportReason::EXPORT_BY_MU && mu_reason == feta::MuExportReason::ACTIVE_TIMEOUT;
}

void NHTFlowCache::export_external(const Packet& pkt) noexcept
{
   m_ctt_stats.export_packets++;
   if (pkt.packet_len != sizeof(feta::CttExportPkt)) {
      m_ctt_stats.export_packets_parsing_failed++;
      MAYBE_DISABLED_CODE(std::cout << "Parsing external export failed" << std::endl;)
      return;
   }
   MAYBE_DISABLED_CODE(std::vector<std::byte> data(reinterpret_cast<const std::byte*>(pkt.packet), reinterpret_cast<const std::byte*>(pkt.packet) + pkt.packet_len);)
   feta::CttExportPkt export_data = feta::CttExportPkt::deserialize(reinterpret_cast<std::byte*>(const_cast<uint8_t*>(pkt.packet)));
   
   MAYBE_DISABLED_CODE(std::cout << "External export of " << std::hex << export_data.flow_hash << std::endl;)
   const IP ip_version = export_data.record.ip_ver == feta::IpVersion::IPV4 ? IP::v4 : IP::v6;
   ///TODO REMOVE
   /*if (ip_version == IP::v4) {
      std::swap(export_data.record.ip_src[0], export_data.record.ip_src[3]);
      std::swap(export_data.record.ip_dst[0], export_data.record.ip_dst[3]); 
   }*/

   const uint16_t vlan_id = export_data.record.vlan_vld ? export_data.record.vlan_tci & 0x0FFF : 0; 
   const auto [key, swapped] = FlowKeyFactory::create_sorted_key(export_data.record.ip_src.data(), export_data.record.ip_dst.data(),
      export_data.record.port_src, export_data.record.port_dst, export_data.record.l4_proto, ip_version, vlan_id);
   /*std::visit([](auto& key) {
      std::reverse(key.src_ip.data(), key.src_ip.data() + sizeof(key.src_ip));
      std::reverse(key.dst_ip.data(), key.dst_ip.data() + sizeof(key.dst_ip));
   }, key);*/
   const auto [search, source_to_destination] = find_flow_index(key, swapped);
   const auto [row, flow_index, hash_value] = search;
   if (!flow_index.has_value()
         || !m_flow_table[flow_index.value()]->is_in_ctt
         || !m_flow_table[flow_index.value()]->offload_mode.has_value()) {
      m_ctt_stats.export_packets_for_missing_flow++;
      MAYBE_DISABLED_CODE(std::cout << "Export of missing flow" << std::endl;)
      return;
   }

   update_ctt_export_stats(export_data.fields.rsn, export_data.fields.ursn);

   if (m_flow_table[flow_index.value()]->offload_mode == feta::OffloadMode::DROP_PACKET_DROP_META) {
      m_flow_table[flow_index.value()]->is_waiting_ctt_response = false;
      //if (is_active_timeout(export_data.fields.rsn, export_data.fields.ursn))
         update_packet_counters_from_active_timeout_external_export(m_flow_table[flow_index.value()]->m_flow, export_data.record);
      //else
      //   update_packet_counters_from_external_export(m_flow_table[flow_index.value()]->m_flow, export_data.record);
      MAYBE_DISABLED_CODE(std::cout << "Update of 3L offloaded flow" << std::endl;)  
   }

   if (!export_data.fields.wb 
      && m_flow_table[flow_index.value()]->offload_mode == feta::OffloadMode::TRIM_PACKET_META) {
      m_flow_table[flow_index.value()]->is_waiting_ctt_response = false;
   }

   MAYBE_DISABLED_CODE(std::cout << "Write back =" << uint32_t(export_data.fields.wb) << std::endl;)

   if (!export_data.fields.wb  
      && m_flow_table[flow_index.value()]->offload_mode == feta::OffloadMode::DROP_PACKET_DROP_META 
      && (is_tcp_restart(export_data.fields.rsn, export_data.fields.ursn)
         || is_active_timeout(export_data.fields.rsn, export_data.fields.ursn)
         || export_data.fields.rsn == feta::ExportReason::EXPORT_BY_SW)) {
      MAYBE_DISABLED_CODE(std::cout << "Real export of ctt site FULL OFFLOADED flow from the ipfixprobe because of TCP/AT/SW" << std::endl;)
      export_flow(flow_index.value());
      m_ctt_stats.flows_removed++;
      return;
   }
   
   if (!export_data.fields.wb && m_flow_table[flow_index.value()]->offload_mode == feta::OffloadMode::TRIM_PACKET_META) {
      MAYBE_DISABLED_CODE(std::cout << "Real export of ctt site TRIMMED flow from the ipfixprobe because of SW" << std::endl;)
      export_flow(flow_index.value());
      m_ctt_stats.flows_removed++;
      return;
   }
   /*   if (!export_data.fields.wb 
         && m_flow_table[flow_index.value()]->is_waiting_ctt_response 
         && m_flow_table[flow_index.value()]->offload_mode == feta::OffloadMode::TRIM_PACKET_META) {
      std::cout << "Real export of ctt site flow from the ipfixprobe" << std::endl;
      m_flow_table[flow_index.value()]->is_waiting_ctt_response = false;
      export_flow(flow_index.value(), convert_ctt_export_reason_to_ipfxiprobe(export_data.fields.rsn, export_data.fields.ursn));
      m_ctt_stats.flows_removed++;
   }*/

   if (!export_data.fields.wb) {
      m_flow_table[flow_index.value()]->is_in_ctt = false;
      m_flow_table[flow_index.value()]->offload_mode = std::nullopt;
   }

   if (is_counter_overflow(export_data.fields.rsn, export_data.fields.ursn)) {
      MAYBE_DISABLED_CODE(std::cout << "Counter overflow" << std::endl;)
   }

   if (is_hash_collision(export_data.fields.rsn, export_data.fields.ursn)) {
      m_flow_table[flow_index.value()]->can_be_offloaded = false;
      MAYBE_DISABLED_CODE(std::cout << "Found hash collision, disabling flow offload" << std::endl;)
   }

   if (export_data.fields.rsn == feta::ExportReason::FULL_CTT) {
      m_flow_table[flow_index.value()]->can_be_offloaded = false;
      MAYBE_DISABLED_CODE(std::cout << "CTT is full" << std::endl;)
   }

   
}
#endif /* WITH_CTT */

static bool check_ip_version(const Packet& pkt) noexcept
{
   return pkt.ip_version == IP::v4 || pkt.ip_version == IP::v6;
}

int NHTFlowCache::put_pkt(Packet& packet)
{
   MAYBE_DISABLED_CODE(std::vector<char> data(packet.packet, packet.packet + packet.packet_len);)
   plugins_pre_create(packet);

   if (m_enable_fragmentation_cache) {
      try_to_fill_ports_to_fragmented_packet(packet);
   }

   if (!check_ip_version(packet)) {
      return 0;
   }
   m_cache_stats.total++;
   /*const FlowKey direct_key = FlowKeyFactory::create_direct_key(&packet.src_ip, &packet.dst_ip,
      packet.src_port, packet.dst_port, packet.ip_proto, static_cast<IP>(packet.ip_version), packet.vlan_id);
   const FlowKey reversed_key = FlowKeyFactory::create_reversed_key(&packet.src_ip, &packet.dst_ip,
      packet.src_port, packet.dst_port, packet.ip_proto, static_cast<IP>(packet.ip_version), packet.vlan_id);*/

   const auto [sorted_key, swapped] = FlowKeyFactory::create_sorted_key(&packet.src_ip, &packet.dst_ip,
      packet.src_port, packet.dst_port, packet.ip_proto, static_cast<IP>(packet.ip_version), packet.vlan_id);

   prefetch_export_expired();

   auto [flow_search, source_to_destination] =
      //find_flow_index(direct_key, reversed_key);
      find_flow_index(sorted_key, swapped);

   packet.source_pkt = flow_search.flow_index.has_value() ? source_to_destination : true;

   if (!flow_search.flow_index.has_value()) {
      const size_t empty_place = get_empty_place(flow_search.cache_row, packet.ts) + (flow_search.hash_value & m_line_mask);
      create_record(packet, empty_place, flow_search.hash_value);
      m_flow_table[empty_place]->m_flow.swapped = source_to_destination;
      export_expired(packet.ts);
      return 0;
   }

   size_t flow_index = *flow_search.flow_index;
   /*if (std::memcmp(&m_flow_table[flow_index]->m_flow.src_ip, &packet.src_ip, 4) != 0) {
      m_cache_stats.bad++;
   }*/
   if (m_flow_table[flow_index]->m_flow.src_port == 0 || m_flow_table[flow_index]->m_flow.dst_port == 0) {
      m_cache_stats.bad++;
   }  

#ifdef WITH_CTT
   const bool flow_is_waiting_for_export = m_flow_table[flow_index]->is_waiting_ctt_response;
   //const bool flow_is_waiting_for_export = !try_to_export_delayed_flow(packet, flow_index) && m_flow_table[flow_index]->is_waiting_ctt_response;
#else
   constexpr bool flow_is_waiting_for_export = false;
#endif /* WITH_CTT */

#ifdef WITH_CTT
   /*if (m_flow_table[flow_index]->is_empty()) {
      create_record(packet, flow_index, flow_search.hash_value);
      export_expired(packet.ts);
      return 0;
   }*/
#endif /* WITH_CTT */

   /* Existing flow record was found, put flow record at the first index of flow line. */
   const size_t relative_flow_index = flow_index % m_line_size;
   m_cache_stats.lookups += relative_flow_index + 1;
   m_cache_stats.lookups2 += (relative_flow_index + 1) * (relative_flow_index + 1);
   m_cache_stats.hits++;

   flow_search.cache_row.advance_flow(relative_flow_index);
   return update_flow(packet, flow_index - relative_flow_index, flow_is_waiting_for_export);
}

size_t NHTFlowCache::get_empty_place(CacheRowSpan& row, const timeval& now) noexcept
{
   if (const std::optional<size_t> empty_index = row.find_empty(); empty_index.has_value()) {
      m_cache_stats.empty++;
      m_cache_stats.empty_places[empty_index.value()]++;
      return empty_index.value();
   }
   m_cache_stats.not_empty++;

#ifdef WITH_CTT
   const size_t victim_index = row.find_victim(now);
#else /* WITH_CTT */
   const size_t victim_index = m_line_size - 1;
#endif /* WITH_CTT */
   row.advance_flow_to(victim_index, m_new_flow_insert_index);
#ifdef WITH_CTT
   /*if (row[m_new_flow_insert_index]->is_in_ctt && !row[m_new_flow_insert_index]->is_waiting_ctt_response) {
      row[m_new_flow_insert_index]->is_waiting_ctt_response = true;
      if (!row[m_new_flow_insert_index]->offload_mode.has_value()) {
         std::cout << "Flow is in CTT but offload mode is not set" << std::endl;
      }
      if (row[m_new_flow_insert_index]->offload_mode == feta::OffloadMode::DROP_PACKET_DROP_META) {
         m_ctt_controller->remove_record_without_notification(row[m_new_flow_insert_index]->m_flow.flow_hash_ctt);
      } else if (row[m_new_flow_insert_index]->offload_mode == feta::OffloadMode::TRIM_PACKET_META) {
         m_ctt_controller->export_record(row[m_new_flow_insert_index]->m_flow.flow_hash_ctt);
      }
      row[m_new_flow_insert_index]->last_request_time = now; //{now.tv_sec + 1, now.tv_usec};
   }*/
   if (row[m_new_flow_insert_index]->is_in_ctt) {
      m_ctt_stats.total_requests_count++;
      m_ctt_controller->remove_record_without_notification(row[m_new_flow_insert_index]->m_flow.flow_hash_ctt);
   }
#endif /* WITH_CTT */
   plugins_pre_export(row[m_new_flow_insert_index]->m_flow);
   export_flow(&row[m_new_flow_insert_index] - m_flow_table.data(), FLOW_END_NO_RES);
   return m_new_flow_insert_index;
}

bool NHTFlowCache::try_to_export_on_active_timeout(size_t flow_index, const timeval& now) noexcept
{
   if (!m_flow_table[flow_index]->is_empty() && now.tv_sec - m_flow_table[flow_index]->m_flow.time_first.tv_sec >= m_active) {
      return try_to_export(flow_index, true, now, FLOW_END_ACTIVE);
   }
   return false;
}

void NHTFlowCache::try_to_fill_ports_to_fragmented_packet(Packet& packet)
{
   m_fragmentation_cache.process_packet(packet);
}

uint8_t NHTFlowCache::get_export_reason(const Flow& flow)
{
   constexpr uint8_t TCP_FIN = 0x01;
   constexpr uint8_t TCP_RST = 0x04;
   if ((flow.src_tcp_flags | flow.dst_tcp_flags) & (TCP_FIN | TCP_RST)) {
      // When FIN or RST is set, TCP connection ended naturally
      return FLOW_END_EOF;
   }
   return FLOW_END_INACTIVE;
}

void NHTFlowCache::export_expired(time_t now)
{
   export_expired({now, 0});
}

void NHTFlowCache::export_expired(const timeval& now)
{
   for (size_t i = m_last_exported_on_timeout_index; i < m_last_exported_on_timeout_index + m_new_flow_insert_index; i++) {
      try_to_export_on_inactive_timeout(i, now);
   }
   m_last_exported_on_timeout_index = (m_last_exported_on_timeout_index + m_new_flow_insert_index) & (m_cache_size - 1);
}

void NHTFlowCache::print_report() const
{
   const float tmp = static_cast<float>(m_cache_stats.lookups) / m_cache_stats.hits;
   std::cout << "Total: " << m_cache_stats.total << "\n";
   std::cout << "Hits: " << m_cache_stats.hits << "\n";
   std::cout << "Empty: " << m_cache_stats.empty << "\n";
   std::cout << "Not empty: " << m_cache_stats.not_empty << "\n";
   std::cout << "Expired: " << m_cache_stats.exported << "\n";
   std::cout << "Flushed: " << m_cache_stats.flushed << "\n";
   std::cout << "Average Lookup:  " << tmp << "\n";
   std::cout << "Variance Lookup: " << static_cast<float>(m_cache_stats.lookups2) / m_cache_stats.hits - tmp * tmp << "\n";
   std::cout << "Bad: " << m_cache_stats.bad << "\n";
   for (size_t i = 0; i < m_line_size; i++) {
      std::cout << "Empty places in line " << i << ": " << m_cache_stats.empty_places[i] << "\n";
   }
#ifdef WITH_CTT
   std::cout << "Really processed: " << m_ctt_stats.real_processed_packets << "\n";
   std::cout << "CTT offloaded: " << m_ctt_stats.flows_offloaded << "\n";
   std::cout << "CTT trim packet offloaded: " << m_ctt_stats.trim_packet_offloaded << "\n";
   std::cout << "CTT drop packet offloaded: " << m_ctt_stats.drop_packet_offloaded << "\n";
   std::cout << "CTT flows removed after export packet: " << m_ctt_stats.flows_removed << "\n";
   std::cout << "CTT sent export packets:" << m_ctt_stats.export_packets << "\n";
   std::cout << "CTT export packets parsing failed:" << m_ctt_stats.export_packets_parsing_failed << "\n";
   std::cout << "CTT export packet failed to find corresponding flow:" << m_ctt_stats.export_packets_for_missing_flow << "\n";
   std::cout << "CTT export reasons: " << "\n";
   std::cout << "CTT exports by ipfixprobe request: " << m_ctt_stats.export_reasons.by_request << "\n";
   std::cout << "CTT exports if CTT full: " << m_ctt_stats.export_reasons.ctt_full << "\n";
   std::cout << "CTT exports with RESERVED reason: " << m_ctt_stats.export_reasons.reserved << "\n";
   std::cout << "CTT exports with counter overflow reason: " << m_ctt_stats.export_reasons.counter_overflow << "\n";
   std::cout << "CTT exports with TCP EOF reason: " << m_ctt_stats.export_reasons.tcp_eof << "\n";
   std::cout << "CTT exports with active timeout reason: " << m_ctt_stats.export_reasons.active_timeout << "\n";
   std::cout << "CTT total requests count: " << m_ctt_stats.total_requests_count << "\n";
   std::cout << "CTT lost requests count: " << m_ctt_stats.lost_requests_count << "\n";
   std::cout << "CTT close packet after offload: " << m_ctt_stats.packet_right_after_offload << "\n";
#endif /* WITH_CTT */
}

void NHTFlowCache::set_telemetry_dir(std::shared_ptr<telemetry::Directory> dir)
{
   telemetry::FileOps statsOps = {[=]() { return get_cache_telemetry(); }, nullptr};
   register_file(dir, "cache-stats", statsOps);

   if (m_enable_fragmentation_cache) {
      m_fragmentation_cache.set_telemetry_dir(dir);
   }
}

void NHTFlowCache::update_flow_record_stats(uint64_t packets_count)
{
   if (packets_count == 1) {
      m_flow_record_stats.packets_count_1++;
   } else if (packets_count >= 2 && packets_count <= 5) {
      m_flow_record_stats.packets_count_2_5++;
   } else if (packets_count >= 6 && packets_count <= 10) {
      m_flow_record_stats.packets_count_6_10++;
   } else if (packets_count >= 11 && packets_count <= 20) {
      m_flow_record_stats.packets_count_11_20++;
   } else if (packets_count >= 21 && packets_count <= 50) {
      m_flow_record_stats.packets_count_21_50++;
   } else {
      m_flow_record_stats.packets_count_51_plus++;
   }
}

void NHTFlowCache::update_flow_end_reason_stats(uint8_t reason)
{
   switch (reason) {
   case FLOW_END_ACTIVE:
      m_flow_end_reason_stats.active_timeout++;
      break;
   case FLOW_END_INACTIVE:
      m_flow_end_reason_stats.inactive_timeout++;
      break;
   case FLOW_END_EOF:
      m_flow_end_reason_stats.end_of_flow++;
      break;
   case FLOW_END_NO_RES:
      m_flow_end_reason_stats.collision++;
      break;
   case FLOW_END_FORCED:
      m_flow_end_reason_stats.forced++;
      break;
   default:
      break;
   }
}

telemetry::Content NHTFlowCache::get_cache_telemetry()
{
   telemetry::Dict dict;

   dict["FlowEndReason:ActiveTimeout"] = m_flow_end_reason_stats.active_timeout;
   dict["FlowEndReason:InactiveTimeout"] = m_flow_end_reason_stats.inactive_timeout;
   dict["FlowEndReason:EndOfFlow"] = m_flow_end_reason_stats.end_of_flow;
   dict["FlowEndReason:Collision"] = m_flow_end_reason_stats.collision;
   dict["FlowEndReason:Forced"] = m_flow_end_reason_stats.forced;

   dict["FlowsInCache"] = m_cache_stats.flows_in_cache;
   dict["FlowCacheUsage"] = telemetry::ScalarWithUnit {double(m_cache_stats.flows_in_cache) / m_cache_size * 100, "%"};

   dict["FlowRecordStats:1packet"] = m_flow_record_stats.packets_count_1;
   dict["FlowRecordStats:2-5packets"] = m_flow_record_stats.packets_count_2_5;
   dict["FlowRecordStats:6-10packets"] = m_flow_record_stats.packets_count_6_10;
   dict["FlowRecordStats:11-20packets"] = m_flow_record_stats.packets_count_11_20;
   dict["FlowRecordStats:21-50packets"] = m_flow_record_stats.packets_count_21_50;
   dict["FlowRecordStats:51-plusPackets"] = m_flow_record_stats.packets_count_51_plus;

   dict["TotalExportedFlows"] = m_cache_stats.total_exported;
#ifdef WITH_CTT
   dict["CttRequests"] = m_ctt_stats.total_requests_count;
#endif /* WITH_CTT */
   return dict;
}

void NHTFlowCache::prefetch_export_expired() const
{
   for (decltype(m_last_exported_on_timeout_index) i = m_last_exported_on_timeout_index; i < m_last_exported_on_timeout_index + m_new_flow_insert_index; i++) {
      __builtin_prefetch(m_flow_table[i], 0, 1);
   }
}
#ifdef WITH_CTT
void NHTFlowCache::set_ctt_config(const std::shared_ptr<CttController>& ctt_controller, uint8_t dma_channel)
{
   m_ctt_controller = ctt_controller;
   m_dma_channel = dma_channel;
}
#endif /* WITH_CTT */

}
