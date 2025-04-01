#pragma once 

#include "parser.cuh"
#include "../nfb/src/ndp.hpp"
#include <ipfixprobe/parser-stats.hpp>

namespace ipxp {


void parse_burst_gpu(
    parser_opt_t* opt,
    ParserStats* stats,
    struct ndp_packet* buffer,
    size_t buffer_size);

}