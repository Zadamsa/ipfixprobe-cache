#pragma once

#include <cstddef>


struct FlowHash{
    size_t direct;
    size_t reverse;
};

extern "C" void hash_burst_gpu(struct Packet* buffer, size_t buffer_size, struct FlowHash* hashes);