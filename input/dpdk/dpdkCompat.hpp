/**
 * \file
 * \brief Compatible definitions for DPDK versions.
 * \author Pavel Siska <siska@cesnet.cz>
 * \date 2024
 */
/*
 * Copyright (C) 2024 CESNET
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

#include <rte_ethdev.h>
#include <rte_version.h>

#if RTE_VERSION < RTE_VERSION_NUM(22, 0, 0, 0)
#define RTE_ETH_MQ_RX_RSS ETH_MQ_RX_RSS
#endif

#if RTE_VERSION < RTE_VERSION_NUM(21, 11, 0, 0)
#define RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE DEV_TX_OFFLOAD_MBUF_FAST_FREE

#define RTE_ETH_RX_OFFLOAD_CHECKSUM DEV_RX_OFFLOAD_CHECKSUM

#define RTE_ETH_RX_OFFLOAD_VLAN_STRIP       DEV_RX_OFFLOAD_VLAN_STRIP
#define RTE_ETH_RX_OFFLOAD_IPV4_CKSUM       DEV_RX_OFFLOAD_IPV4_CKSUM
#define RTE_ETH_RX_OFFLOAD_UDP_CKSUM        DEV_RX_OFFLOAD_UDP_CKSUM
#define RTE_ETH_RX_OFFLOAD_TCP_CKSUM        DEV_RX_OFFLOAD_TCP_CKSUM
#define RTE_ETH_RX_OFFLOAD_TCP_LRO          DEV_RX_OFFLOAD_TCP_LRO
#define RTE_ETH_RX_OFFLOAD_QINQ_STRIP       DEV_RX_OFFLOAD_QINQ_STRIP
#define RTE_ETH_RX_OFFLOAD_OUTER_IPV4_CKSUM DEV_RX_OFFLOAD_OUTER_IPV4_CKSUM
#define RTE_ETH_RX_OFFLOAD_MACSEC_STRIP     DEV_RX_OFFLOAD_MACSEC_STRIP
#define RTE_ETH_RX_OFFLOAD_HEADER_SPLIT     DEV_RX_OFFLOAD_HEADER_SPLIT
#define RTE_ETH_RX_OFFLOAD_VLAN_FILTER      DEV_RX_OFFLOAD_VLAN_FILTER
#define RTE_ETH_RX_OFFLOAD_VLAN_EXTEND      DEV_RX_OFFLOAD_VLAN_EXTEND
#define RTE_ETH_RX_OFFLOAD_SCATTER          DEV_RX_OFFLOAD_SCATTER
#define RTE_ETH_RX_OFFLOAD_TIMESTAMP        DEV_RX_OFFLOAD_TIMESTAMP
#define RTE_ETH_RX_OFFLOAD_SECURITY         DEV_RX_OFFLOAD_SECURITY
#define RTE_ETH_RX_OFFLOAD_KEEP_CRC         DEV_RX_OFFLOAD_KEEP_CRC
#define RTE_ETH_RX_OFFLOAD_SCTP_CKSUM       DEV_RX_OFFLOAD_SCTP_CKSUM
#define RTE_ETH_RX_OFFLOAD_OUTER_UDP_CKSUM  DEV_RX_OFFLOAD_OUTER_UDP_CKSUM
#define RTE_ETH_RX_OFFLOAD_RSS_HASH DEV_RX_OFFLOAD_RSS_HASH

#define RTE_ETH_MQ_TX_NONE ETH_MQ_TX_NONE

#define RTE_ETH_MQ_RX_NONE ETH_MQ_RX_NONE

#define RTE_ETH_RSS_IP     ETH_RSS_IP
#define RTE_ETH_RSS_UDP    ETH_RSS_UDP
#define RTE_ETH_RSS_TCP    ETH_RSS_TCP
#define RTE_ETH_RSS_SCTP   ETH_RSS_SCTP
#define RTE_ETH_RSS_TUNNEL ETH_RSS_TUNNEL

#define RTE_ETH_RSS_L3_SRC_ONLY ETH_RSS_L3_SRC_ONLY
#define RTE_ETH_RSS_L3_DST_ONLY ETH_RSS_L3_DST_ONLY
#define RTE_ETH_RSS_L4_SRC_ONLY ETH_RSS_L4_SRC_ONLY
#define RTE_ETH_RSS_L4_DST_ONLY ETH_RSS_L4_DST_ONLY

#define RTE_ETH_RSS_IPV4               ETH_RSS_IPV4
#define RTE_ETH_RSS_FRAG_IPV4          ETH_RSS_FRAG_IPV4
#define RTE_ETH_RSS_NONFRAG_IPV4_TCP   ETH_RSS_NONFRAG_IPV4_TCP
#define RTE_ETH_RSS_NONFRAG_IPV4_UDP   ETH_RSS_NONFRAG_IPV4_UDP
#define RTE_ETH_RSS_NONFRAG_IPV4_SCTP  ETH_RSS_NONFRAG_IPV4_SCTP
#define RTE_ETH_RSS_NONFRAG_IPV4_OTHER ETH_RSS_NONFRAG_IPV4_OTHER
#define RTE_ETH_RSS_IPV6               ETH_RSS_IPV6
#define RTE_ETH_RSS_FRAG_IPV6          ETH_RSS_FRAG_IPV6
#define RTE_ETH_RSS_NONFRAG_IPV6_TCP   ETH_RSS_NONFRAG_IPV6_TCP
#define RTE_ETH_RSS_NONFRAG_IPV6_UDP   ETH_RSS_NONFRAG_IPV6_UDP
#define RTE_ETH_RSS_NONFRAG_IPV6_SCTP  ETH_RSS_NONFRAG_IPV6_SCTP
#define RTE_ETH_RSS_NONFRAG_IPV6_OTHER ETH_RSS_NONFRAG_IPV6_OTHER
#define RTE_ETH_RSS_L2_PAYLOAD         ETH_RSS_L2_PAYLOAD
#define RTE_ETH_RSS_IPV6_EX            ETH_RSS_IPV6_EX
#define RTE_ETH_RSS_IPV6_TCP_EX        ETH_RSS_IPV6_TCP_EX
#define RTE_ETH_RSS_IPV6_UDP_EX        ETH_RSS_IPV6_UDP_EX
#define RTE_ETH_RSS_PORT               ETH_RSS_PORT
#define RTE_ETH_RSS_VXLAN              ETH_RSS_VXLAN
#define RTE_ETH_RSS_NVGRE              ETH_RSS_NVGRE
#define RTE_ETH_RSS_GTPU               ETH_RSS_GTPU
#define RTE_ETH_RSS_GENEVE             ETH_RSS_GENEVE

#define RTE_BIT64(nr)   (UINT64_C(1) << (nr))

#endif