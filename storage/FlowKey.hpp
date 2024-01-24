//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_FLOW_KEY_H
#define IPFIXPROBE_CACHE_FLOW_KEY_H

namespace ipxp {

template<uint16_t IPSize>
struct __attribute__((packed)) FlowKey {
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t proto;
    uint8_t ip_version;
    std::array<uint8_t, IPSize> src_ip;
    std::array<uint8_t, IPSize> dst_ip;
    uint16_t vlan_id;
    flow_key<IPSize>& operator=(const Packet& pkt) noexcept;
    flow_key<IPSize>& save_reversed(const Packet& pkt) noexcept;
};

} // namespace ipxp
#endif // IPFIXPROBE_CACHE_FLOW_KEY_H
