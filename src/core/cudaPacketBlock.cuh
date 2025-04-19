#pragma once

#include <ipfixprobe/packet.hpp>

namespace ipxp {

PacketBlock* getCudaPacketBlock(size_t pkts_size);

void freeCudaPacketBlock(PacketBlock* block);

} // namespace ipxp