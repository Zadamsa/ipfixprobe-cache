#pragma once

#include <cstddef>

#include <ipfixprobe/packet.hpp>

namespace ipxp{

void hash_burst_gpu(PacketBlock& parsed_packets);

void gpu_haher_init(Packet* packets);

} // namespace ipxp