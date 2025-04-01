#pragma once

#include <cstddef>

namespace ipxp{

__host__ void hash_burst_gpu(struct Packet* buffer, size_t buffer_size, struct FlowHash* hashes);

__host__ void gpu_haher_init();

__host__ void gpu_haher_close();

} // namespace ipxp