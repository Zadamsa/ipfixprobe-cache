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

struct PacketData{
    const uint8_t* data;
    size_t length;
    struct timeval ts;
};

void parse_burst_gpu(PacketBlock& parsed_result);
//void parse_burst_gpu(PacketBlock& parsed_result, const std::vector<PacketData>& packets);

void close_gpu_parser();

void init_gpu_parser();

}