/**
* \file
 * \author Damir Zainullin <zaidamilda@gmail.com>
 * \brief FlowRecord implementation.
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

#include "flowRecord.hpp"

#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/packet.hpp>
#include <cstring>

namespace ipxp {

FlowRecord::FlowRecord()
{
   erase();
};

FlowRecord::~FlowRecord()
{
   erase();
};

void FlowRecord::erase()
{
   m_flow.remove_extensions();
   m_hash = 0;
   memset(&m_flow.time_first, 0, sizeof(m_flow.time_first));
   memset(&m_flow.time_last, 0, sizeof(m_flow.time_last));
   m_flow.ip_version = 0;
   m_flow.ip_proto = 0;
   memset(&m_flow.src_ip, 0, sizeof(m_flow.src_ip));
   memset(&m_flow.dst_ip, 0, sizeof(m_flow.dst_ip));
   m_flow.src_port = 0;
   m_flow.dst_port = 0;
   m_flow.src_packets = 0;
   m_flow.dst_packets = 0;
   m_flow.src_bytes = 0;
   m_flow.dst_bytes = 0;
   m_flow.src_tcp_flags = 0;
   m_flow.dst_tcp_flags = 0;
#ifdef WITH_CTT
   is_waiting_ctt_response = false;
   is_in_ctt = false;
   last_request_time = {0, 0};
   can_be_offloaded = false;
   offload_mode = std::nullopt;
#endif /* WITH_CTT */
}
void FlowRecord::reuse()
{
   m_flow.remove_extensions();
   m_flow.time_first = m_flow.time_last;
   m_flow.src_packets = 0;
   m_flow.dst_packets = 0;
   m_flow.src_bytes = 0;
   m_flow.dst_bytes = 0;
   m_flow.src_tcp_flags = 0;
   m_flow.dst_tcp_flags = 0;
#ifdef WITH_CTT
   is_waiting_ctt_response = false;
   is_in_ctt = false;
   last_request_time = {0, 0};
   can_be_offloaded = false;
   offload_mode = std::nullopt;
#endif /* WITH_CTT */
}

void FlowRecord::create(const Packet &pkt, uint64_t hash)
{
   m_flow.src_packets = 1;

   m_hash = hash;

   m_flow.time_first = pkt.ts;
   m_flow.time_last = pkt.ts;
   m_flow.flow_hash = hash;
   m_flow.vlan_id = pkt.vlan_id;

   memcpy(m_flow.src_mac, pkt.src_mac, 6);
   memcpy(m_flow.dst_mac, pkt.dst_mac, 6);

   if (pkt.ip_version == IP::v4) {
      m_flow.ip_version = pkt.ip_version;
      m_flow.ip_proto = pkt.ip_proto;
      m_flow.src_ip.v4 = pkt.src_ip.v4;
      m_flow.dst_ip.v4 = pkt.dst_ip.v4;
      m_flow.src_bytes = pkt.ip_len;
   } else if (pkt.ip_version == IP::v6) {
      m_flow.ip_version = pkt.ip_version;
      m_flow.ip_proto = pkt.ip_proto;
      memcpy(m_flow.src_ip.v6, pkt.src_ip.v6, 16);
      memcpy(m_flow.dst_ip.v6, pkt.dst_ip.v6, 16);
      m_flow.src_bytes = pkt.ip_len;
   }

   if (pkt.ip_proto == IPPROTO_TCP) {
      m_flow.src_port = pkt.src_port;
      m_flow.dst_port = pkt.dst_port;
      m_flow.src_tcp_flags = pkt.tcp_flags;
   } else if (pkt.ip_proto == IPPROTO_UDP) {
      m_flow.src_port = pkt.src_port;
      m_flow.dst_port = pkt.dst_port;
   } else if (pkt.ip_proto == IPPROTO_ICMP ||
      pkt.ip_proto == IPPROTO_ICMPV6) {
      m_flow.src_port = pkt.src_port;
      m_flow.dst_port = pkt.dst_port;
   }
#ifdef WITH_CTT
   is_waiting_ctt_response = false;
   is_in_ctt = false;
   can_be_offloaded = true;
   last_request_time = {pkt.ts.tv_sec - 1, pkt.ts.tv_usec};
#endif /* WITH_CTT */
}

void FlowRecord::update(const Packet &pkt)
{
   m_flow.time_last = pkt.ts;
   if (pkt.source_pkt) {
      m_flow.src_packets++;
      m_flow.src_bytes += pkt.ip_len;

      if (pkt.ip_proto == IPPROTO_TCP) {
         m_flow.src_tcp_flags |= pkt.tcp_flags;
      }
   } else {
      m_flow.dst_packets++;
      m_flow.dst_bytes += pkt.ip_len;

      if (pkt.ip_proto == IPPROTO_TCP) {
         m_flow.dst_tcp_flags |= pkt.tcp_flags;
      }
   }
#ifdef WITH_CTT

#endif /* WITH_CTT */
}

} // ipxp