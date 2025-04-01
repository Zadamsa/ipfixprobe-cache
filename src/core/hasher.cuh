#pragma once

#include <cstddef>

namespace ipxp{

struct FlowHash{
    size_t direct;
    size_t reverse;
};

void hash_burst_gpu(struct Packet* buffer, size_t buffer_size, struct FlowHash* hashes);

} // namespace ipxp