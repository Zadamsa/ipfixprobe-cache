//
// Created by zaida on 24.01.2024.
//

#include "FlowKeyV4.hpp"

namespace ipxp {
FlowKeyV4& FlowKeyV4::operator=(const Packet& pkt) noexcept
{
    flow_key::operator=(pkt);
    ip_version = IP::v4;
    memcpy(src_ip.data(), &pkt.src_ip.v4, 4);
    memcpy(dst_ip.data(), &pkt.dst_ip.v4, 4);
    return *this;
}

FlowKeyV4& FlowKeyV4::save_reversed(const Packet& pkt) noexcept
{
    flow_key::save_reversed(pkt);
    ip_version = IP::v4;
    memcpy(src_ip.data(), &pkt.dst_ip.v4, 4);
    memcpy(dst_ip.data(), &pkt.src_ip.v4, 4);
    return *this;
}
} // namespace ipxp