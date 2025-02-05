/**
* \file
 * \author Damir Zainullin <zaidamilda@gmail.com>
 * \brief FlowKey structure declaration.
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

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>

namespace ipxp {

template<IP Version>
struct FlowKey {
   static constexpr size_t AddressSize = Version == IP::v4 ? 4 : 16;
   uint16_t src_port;
   uint16_t dst_port;
   uint8_t proto;
   uint8_t ip_version;
   std::array<uint8_t, AddressSize> src_ip;
   std::array<uint8_t, AddressSize> dst_ip;
} __attribute__((packed));

using FlowKeyv4 = FlowKey<IP::v4>;
using FlowKeyv6 = FlowKey<IP::v6>;

} // namespace ipxp