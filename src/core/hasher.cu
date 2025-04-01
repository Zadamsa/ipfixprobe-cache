#include "hasher.cuh"
#include "xxhash.h"
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <ipfixprobe/packet.hpp>
#include <ipfixprobe/storagePlugin.hpp>


namespace ipxp{

FlowHash* hashes;

struct Key {
	uint8_t src_ip[16];
	uint8_t dst_ip[16];
	uint16_t src_port;
	uint16_t dst_port;
	uint8_t protocol;
	uint8_t ipv;
} __attribute__((packed));

Key* keys;
Key* keys_reversed;
__global__ void hash(Key* keys, Key* keys_reversed, FlowHash* result_hashes, size_t buffer_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= buffer_size) {
		return;
	}
	result_hashes[idx] = {XXH64(&keys[idx], sizeof(Key), 0), XXH64(&keys_reversed[idx], sizeof(Key), 0)};
}

void hash_burst_gpu(struct Packet* buffer, size_t buffer_size, struct FlowHash* result_hashes)
{
	std::for_each_n(buffer, buffer_size, [&, index = 0](const Packet& packet) mutable {
		if (packet.ip_version != 4 && packet.ip_version != 6) {
			return;
		}
		cudaMemset(&keys[index].src_ip, 0, sizeof(keys[index].src_ip));
		cudaMemset(&keys_reversed[index].src_ip, 0, sizeof(keys_reversed[index].src_ip));
		cudaMemset(&keys[index].dst_ip, 0, sizeof(keys[index].dst_ip));
		cudaMemset(&keys_reversed[index].dst_ip, 0, sizeof(keys_reversed[index].dst_ip));
		if (packet.ip_version == 4) {
			cudaMemcpy((void*)(&buffer[index].src_ip), &keys[index].src_ip, 4, cudaMemcpyHostToDevice);
			cudaMemcpy((void*)(&buffer[index].dst_ip), &keys[index].dst_ip, 4, cudaMemcpyHostToDevice);
			cudaMemcpy((void*)(&buffer[index].src_ip), &keys_reversed[index].dst_ip, 4, cudaMemcpyHostToDevice);
			cudaMemcpy((void*)(&buffer[index].dst_ip), &keys_reversed[index].src_ip, 4, cudaMemcpyHostToDevice);
		} else {
			cudaMemcpy((void*)(&buffer[index].src_ip), &keys[index].src_ip, 16, cudaMemcpyHostToDevice);
			cudaMemcpy((void*)(&buffer[index].dst_ip), &keys[index].dst_ip, 16, cudaMemcpyHostToDevice);
			cudaMemcpy((void*)(&buffer[index].src_ip), &keys_reversed[index].dst_ip, 16, cudaMemcpyHostToDevice);
			cudaMemcpy((void*)(&buffer[index].dst_ip), &keys_reversed[index].src_ip, 16, cudaMemcpyHostToDevice);
		}
		cudaMemcpy((void*)(&buffer[index].src_port), &keys[index].src_port, 2, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)(&buffer[index].src_port), &keys_reversed[index].dst_port, 2, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)(&buffer[index].dst_port), &keys[index].dst_port, 2, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)(&buffer[index].dst_port), &keys_reversed[index].src_port, 2, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)(&buffer[index].ip_proto), &keys[index].protocol, 1, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)(&buffer[index].ip_version), &keys[index].ipv, 1, cudaMemcpyHostToDevice);
	});

	int blockSize = 4;
	int numBlocks = (buffer_size + blockSize - 1) / buffer_size;
	hash<<<numBlocks, blockSize>>>(keys, keys_reversed, result_hashes, buffer_size);
    
}

void gpu_haher_init()
{
	cudaMalloc((void **)&hashes, sizeof(FlowHash) * 100);
	cudaMalloc((void **)&keys, sizeof(Key) * 100);
}

void gpu_haher_close()
{
	cudaFree(hashes);
	cudaFree(keys);
}

} // namespace ipxp
