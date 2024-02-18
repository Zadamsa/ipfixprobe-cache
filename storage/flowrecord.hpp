/**
 * \file flowrecord.hpp
 * \brief FlowRecord class wraps flow, all manipulations with the flow go through FlowRecord
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

#ifndef IPFIXPROBE_CACHE_FLOWRECORD_HPP
#define IPFIXPROBE_CACHE_FLOWRECORD_HPP
#include <ipfixprobe/flowifc.hpp>
#include <ipfixprobe/packet.hpp>
#include <string>
#include <sys/time.h>
#include "fragmentationCache/timevalUtils.hpp"
namespace ipxp {

class FlowRecord {
    uint64_t m_hash; ///< Hash value of the flow.
public:
    Flow m_flow;
    bool m_swapped;
    bool m_frozen;
    timeval m_interval;
    timeval m_next_packet_time;
    static const constexpr timeval MAXIMAL_WAITING_TIME = timeval{2,0};
    static inline timeval maximum;
    FlowRecord();
    ~FlowRecord();

    void erase();
    void reuse();

    timeval next_packet_arrives_at(const timeval& now) noexcept;
    inline __attribute__((always_inline)) bool is_empty() const { return m_hash == 0; }
    inline __attribute__((always_inline)) bool belongs(uint64_t hash) const
    {
        return hash == m_hash;
    }
    void create(const Packet& pkt, uint64_t pkt_hash, bool key_swapped);
    void update(const Packet& pkt, bool src);
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_FLOWRECORD_HPP
