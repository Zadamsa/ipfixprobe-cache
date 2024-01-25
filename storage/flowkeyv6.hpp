//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_FLOW_KEY_V6_H
#define IPFIXPROBE_CACHE_FLOW_KEY_V6_H

namespace ipxp {

struct __attribute__((packed)) FlowKeyV6 : public FlowKey<16> {
    FlowKeyV6& operator=(const Packet& pkt) noexcept;
    FlowKeyV6& save_reversed(const Packet& pkt) noexcept;
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_FLOW_KEY_V6_H
