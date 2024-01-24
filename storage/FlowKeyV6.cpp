//
// Created by zaida on 24.01.2024.
//

#include "FlowKeyV6.hpp"

namespace ipxp {
FlowKeyV6& FlowKeyV6::operator=(const Packet& pkt) noexcept
{
    flow_key::operator=(pkt);
    ip_version = IP::v6;
    memcpy(src_ip.data(), pkt.src_ip.v6, 16);
    memcpy(dst_ip.data(), pkt.dst_ip.v6, 16);
    return *this;
}

FlowKeyV6& FlowKeyV6::save_reversed(const Packet& pkt) noexcept
{
    flow_key::save_reversed(pkt);
    ip_version = IP::v6;
    memcpy(src_ip.data(), pkt.dst_ip.v6, 16);
    memcpy(dst_ip.data(), pkt.src_ip.v6, 16);
    return *this;
}
} // namespace ipxp