//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_FLOW_KEY_V4_H
#define IPFIXPROBE_CACHE_FLOW_KEY_V4_H

namespace ipxp {

struct __attribute__((packed)) FlowKeyV4 : public FlowKey<4> {
    FlowKeyV4& operator=(const Packet& pkt) noexcept;
    FlowKeyV4& save_reversed(const Packet& pkt) noexcept;
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_FLOW_KEY_V4_H
