//
// Created by zaida on 24.01.2024.
//
#include <cstring>
#include "flowkeyv4.hpp"
#include "flowkey.tpp"

namespace ipxp {
FlowKeyV4& FlowKeyV4::operator=(const Packet& pkt) noexcept
{
    //*(static_cast<FlowKey<4>*>(this)) = pkt;
    FlowKey::operator=(pkt);
    ip_version = IP::v4;
    memcpy(src_ip.data(), &pkt.src_ip.v4, 4);
    memcpy(dst_ip.data(), &pkt.dst_ip.v4, 4);
    return *this;
}

FlowKeyV4& FlowKeyV4::save_reversed(const Packet& pkt) noexcept
{
    FlowKey::save_reversed(pkt);
    ip_version = IP::v4;
    memcpy(src_ip.data(), &pkt.dst_ip.v4, 4);
    memcpy(dst_ip.data(), &pkt.src_ip.v4, 4);
    return *this;
}
} // namespace ipxp