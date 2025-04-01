#pragma once 

#include "parser.cuh"
#include "../nfb/src/ndp.hpp"
#include <ipfixprobe/parser-stats.hpp>

namespace ipxp {

struct PacketData{
    const uint8_t* data;
    size_t length;
    struct timeval ts;
};

void parse_burst_gpu(const std::vector<PacketData>& packets);

}