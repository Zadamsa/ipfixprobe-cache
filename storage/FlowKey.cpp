//
// Created by zaida on 24.01.2024.
//

#include "FlowKey.hpp"

namespace ipxp {
template<uint16_t IPSize>
FlowKey<IPSize>& FlowKey<IPSize>::operator=(const Packet& pkt) noexcept
{
    proto = pkt.ip_proto;
    src_port = pkt.src_port;
    dst_port = pkt.dst_port;
    vlan_id = pkt.vlan_id;
    return *this;
}

template<uint16_t IPSize>
FlowKey<IPSize>& FlowKey<IPSize>::save_reversed(const Packet& pkt) noexcept
{
    *this = pkt;
    src_port = pkt.dst_port;
    dst_port = pkt.src_port;
    return *this;
}
} // namespace ipxp