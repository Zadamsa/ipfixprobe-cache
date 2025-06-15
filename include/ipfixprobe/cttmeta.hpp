#pragma once

#include <cstdint>
#include <optional>
#include <bits/types/struct_timeval.h>
#include <feta.hpp>

namespace ipxp {    

static uint64_t extract(const uint8_t* bitvec, size_t start_bit, size_t bit_length) {
    size_t start_byte = start_bit / 8;
    size_t end_bit = start_bit + bit_length;
    size_t end_byte = (end_bit + 7) / 8;
    uint64_t value = 0;
    for (size_t i = 0; i < end_byte - start_byte; ++i) {
        value |= static_cast<uint64_t>(bitvec[start_byte + i]) << (8 * i);
    }
    value >>= (start_bit % 8);
    uint64_t mask = (bit_length == 64) ? ~0ULL : ((1ULL << bit_length) - 1);
    return value & mask;
}

enum MessageType : uint8_t
{
    FRAME_AND_FULL_METADATA = 0x0, ///< Frame and full metadata
    FRAME_AND_HALF_METADATA = 0x1, ///< Frame and half metadata
    FRAME_WITH_TIMESTAMP    = 0x2, ///< Frame with timestamp
    FRAME_WITH_NO_METADATA  = 0x3, ///< Frame with no metadata
    ONLY_FULL_METADATA      = 0x4, ///< Only full metadata
    FLOW_EXPORT             = 0xF  ///< Flow export
};

enum CsumStatus : uint8_t {
    CSUM_UNKNOWN = 0x0, ///< No information about the checksum
    CSUM_BAD     = 0x1, ///< The checksum in the packet is wrong
    CSUM_GOOD    = 0x2, ///< The checksum in the packet is valid
    CSUM_NONE    = 0x3  ///< Checksum not correct but header integrity verified
};

enum ParserStatus : uint8_t {
    PA_OK      = 0x0, ///< Parsing completed successfully
    PA_UNKNOWN = 0x1, ///< Parser stopped at an unknown protocol
    PA_LIMIT   = 0x2, ///< Parser stopped at its own limit (e.g., VLAN=4)
    PA_ERROR   = 0x3  ///< Error in protocol header or parsing overflow
};

enum L2PType : uint8_t {
    L2_UNKNOWN          = 0x0, ///< Unknown L2 protocol
    L2_ETHER_IP         = 0x1, ///< Ethernet with IP payload
    L2_ETHER_TIMESYNC   = 0x2, ///< Ethernet with TimeSync protocol
    L2_ETHER_ARP        = 0x3, ///< Ethernet with ARP protocol
    L2_ETHER_LLDP       = 0x4, ///< Ethernet with LLDP protocol
    L2_ETHER_NSH        = 0x5, ///< Ethernet with NSH protocol
    L2_ETHER_VLAN       = 0x6, ///< Ethernet with VLAN tagging
    L2_ETHER_QINQ       = 0x7, ///< Ethernet with QinQ tagging
    L2_ETHER_PPPOE      = 0x8, ///< Ethernet with PPPoE encapsulation
    L2_ETHER_FCOE       = 0x9, ///< Ethernet with FCoE protocol
    L2_ETHER_MPLS       = 0xA  ///< Ethernet with MPLS
};

enum L3PType : uint8_t {
    L3_UNKNOWN   = 0x0, ///< Unknown L3 protocol
    L3_IPV4      = 0x1, ///< IPv4 protocol
    L3_IPV4_EXT  = 0x3, ///< IPv4 with extensions
    L3_IPV6      = 0x4, ///< IPv6 protocol
    L3_IPV6_EXT  = 0xC  ///< IPv6 with extensions
};

enum L4PType : uint8_t {
    L4_UNKNOWN = 0x0, ///< Unknown L4 protocol
    L4_TCP     = 0x1, ///< TCP protocol
    L4_UDP     = 0x2, ///< UDP protocol
    L4_FRAG    = 0x3, ///< Fragmented packet
    L4_SCTP    = 0x4, ///< SCTP protocol
    L4_ICMP    = 0x5, ///< ICMP protocol
    L4_NONFRAG = 0x6, ///< Non-fragmented packet
    L4_IGMP    = 0x7  ///< IGMP protocol
};

struct CttMetadata {
    constexpr static size_t SIZE = 32;

    static CttMetadata parse(const uint8_t* data, size_t length) noexcept
    {
        CttMetadata metadata;
        if (length != CttMetadata::SIZE) {
            metadata.flow_hash = 0;
            return metadata;
        }

        //metadata.ts.tv_usec      = extract(data, 0,   32) / 1000; ///< CTT uses seconds/nanoseconds ts
        //metadata.ts.tv_sec       = extract(data, 32,  32);
        //metadata.vlan_tci        = extract(data, 64,  16);
        metadata.vlan_tci = *reinterpret_cast<const uint16_t*>(data + 8);
        //metadata.vlan_vld        = extract(data, 80,  1);
        metadata.vlan_vld = *reinterpret_cast<const uint8_t*>(data + 10) & 0x01;       
        //metadata.vlan_stripped   = extract(data, 81,  1);
        metadata.vlan_stripped = *reinterpret_cast<const uint8_t*>(data + 10) & 0x02;

        //metadata.parser_status   = static_cast<ParserStatus>(extract(data, 86,  2));
        //metadata.parser_status = static_cast<ParserStatus>(*reinterpret_cast<const uint8_t*>(data + 10) & 0xC0);
        //metadata.ifc             = extract(data, 88,  8);
        //metadata.flow_hash       = extract(data, 128, 64);
        metadata.flow_hash = *reinterpret_cast<const uint64_t*>(data + 16);

        return metadata;
        metadata.ip_csum_status  = static_cast<CsumStatus>(extract(data, 82,  2));
        metadata.l4_csum_status  = static_cast<CsumStatus>(extract(data, 84,  2));
        metadata.filter_bitmap   = extract(data, 96,  16);
        metadata.ctt_export_trig = extract(data, 112, 1);
        metadata.ctt_rec_matched = extract(data, 113, 1);
        metadata.ctt_rec_created = extract(data, 114, 1);
        metadata.ctt_rec_deleted = extract(data, 115, 1);
        metadata.l2_len          = extract(data, 192, 7);
        metadata.l3_len          = extract(data, 199, 9);
        metadata.l4_len          = extract(data, 208, 8);
        metadata.l2_ptype        = static_cast<L2PType>(extract(data, 216, 4));
        metadata.l3_ptype        = static_cast<L3PType>(extract(data, 220, 4));
        metadata.l4_ptype        = static_cast<L4PType>(extract(data, 224, 4));
        
        //if (metadata.parser_status != ParserStatus::PA_OK) {
            //return std::nullopt;
        //}
        return metadata;
    }
    struct timeval ts;             ///< Timestamp; invalid if all bits are 1
    uint16_t vlan_tci;             ///< VLAN Tag Control Information from outer VLAN
    bool vlan_vld : 1;             ///< VLAN valid flag; indicates if VLAN TCI is valid
    bool vlan_stripped : 1;        ///< VLAN stripped flag; outer VLAN only
    CsumStatus ip_csum_status : 2; ///< IP checksum status
    CsumStatus l4_csum_status : 2; ///< Layer 4 checksum status
    ParserStatus parser_status : 2;///< Final state of FPGA parser
    uint8_t ifc;                   ///< Interface (IFC) number
    uint16_t filter_bitmap;        ///< Filter bitmap; each filter rule can have several mark bits
    bool ctt_export_trig : 1;      ///< CTT flag; packet triggered export in CTT
    bool ctt_rec_matched : 1;      ///< CTT flag; packet matched record in CTT
    bool ctt_rec_created : 1;      ///< CTT flag; packet created record in CTT
    bool ctt_rec_deleted : 1;      ///< CTT flag; packet deleted record in CTT
    uint64_t flow_hash;            ///< Flow hash; not the same as RSS hash
    uint8_t l2_len : 7;            ///< Length of the L2 layer, if known
    uint16_t l3_len : 9;           ///< Length of the L3 layer, if known
    uint8_t l4_len : 8;            ///< Length of the L4 layer, if known
    L2PType l2_ptype : 4;          ///< Type of the L2 layer
    L3PType l3_ptype : 4;          ///< Type of the L3 layer
    L4PType l4_ptype : 4;          ///< Type of the L4 layer
};

constexpr static timeval CTT_REQUEST_TIMEOUT = {1, 0}; ///< Timeout for CTT request

static constexpr size_t KEY_SIZE = 8;
static constexpr size_t STATE_SIZE = sizeof(feta::CttRecord);
static constexpr size_t MASK_SIZE = 21;


}
