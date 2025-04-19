#pragma once 

#include "parser.cuh"
#include <ipfixprobe/parser-stats.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <ipfixprobe/packet.hpp>
//#include <ipfixprobe/storagePlugin.hpp>

namespace ipxp {

void parse_burst_gpu(PacketBlock& parsed_result);
//void parse_burst_gpu(PacketBlock& parsed_result, const std::vector<PacketData>& packets);

void close_gpu_parser();

void init_gpu_parser(Packet* packets);

}