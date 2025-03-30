/**
* \file
 * \author Damir Zainullin <zaidamilda@gmail.com>
 * \brief CacheRowSpan implementation.
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

#include "cacheRowSpan.hpp"

#include <algorithm>
#include <iostream>
#include "fragmentationCache/timevalUtils.hpp"

namespace ipxp {

CacheRowSpan::CacheRowSpan(FlowRecord** begin, size_t count) noexcept
   : m_begin(begin), m_count(count)
{
}

std::optional<size_t> CacheRowSpan::find_by_hash(uint64_t hash) const noexcept
{
   FlowRecord** it = nullptr;
   it = std::find_if(m_begin, m_begin + m_count, [&](const FlowRecord* flow) {
      return flow->belongs(hash);
   });
   if (it == m_begin + m_count) {
      return std::nullopt;
   }
   return it - m_begin;
}

void CacheRowSpan::advance_flow_to(size_t from, size_t to) noexcept
{
   if (from < to) {
      std::rotate(m_begin + from, m_begin + from + 1, m_begin + to + 1);
      return;
   }
   std::rotate(m_begin + to, m_begin + from, m_begin + from + 1);
}

void CacheRowSpan::advance_flow(size_t flow_index) noexcept
{
   advance_flow_to(flow_index, 0);
}

std::optional<size_t> CacheRowSpan::find_empty() const noexcept
{
   auto it = std::find_if(m_begin, m_begin + m_count, [](const FlowRecord* flow) {
      return flow->is_empty();
   });
   if (it == m_begin + m_count) {
      return std::nullopt;
   }
   return it - m_begin;
}

#ifdef WITH_CTT
size_t CacheRowSpan::find_victim(const timeval& now) const noexcept
{
   FlowRecord* const* victim = m_begin + m_count - 1;
   auto it = std::find_if(std::reverse_iterator(m_begin + m_count), 
                              std::reverse_iterator(m_begin), [&](FlowRecord* const& flow) {
      
      return !flow->is_in_ctt || 
      (flow->offload_mode.has_value() 
         && flow->offload_mode.value() == feta::OffloadMode::TRIM_PACKET_META);
      if (!flow->is_in_ctt || 
         (flow->offload_mode.has_value() 
            && flow->offload_mode.value() == feta::OffloadMode::TRIM_PACKET_META)) {
         return true;
         victim = &flow;
      }
      if (flow->is_in_ctt && !flow->offload_mode.has_value()) {
         std::cout << "Flow is in CTT but offload mode is not set(CRS)" << std::endl;
      }
      return flow->is_waiting_ctt_response 
         && now > flow->last_request_time + CTT_REQUEST_TIMEOUT 
         && flow->offload_mode.has_value();
   });
   if (it == std::reverse_iterator(m_begin)) {
      return m_count - 1;
   }
   return it.base() - m_begin - 1;
}
#endif /* WITH_CTT */

} // ipxp