#include "hasher.cuh"
//#include "xxhash.h"
#include <cstdint>
#include <algorithm>
#include <cstddef>
//#include <ipfixprobe/flowhash.hpp>
#include <array>
#include <cstring>

namespace ipxp{

//FlowHash* hashes;

inline  uint64_t  __device__ fnv1a_hash(const void* data, size_t size) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t __device__ fast_hash(const void* data, size_t len, uint64_t seed = 0xcbf29ce484222325ULL) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t hash = seed;
    while (len--) {
        hash ^= *p++;
        hash *= 0x100000001b3ULL; // FNV prime
    }
    return hash;
}

__forceinline__ __device__  uint32_t super_fast_hash(const char* __restrict__ data, int len) {
    uint32_t hash = len, tmp;
    int rem;

    if (len <= 0 || data == nullptr) return 0;
    rem = len & 3;
    len >>= 2;

    for (; len > 0; len--) {
        hash += *(const uint16_t*)data;
        tmp = (*(const uint16_t*)(data + 2) << 11) ^ hash;
        hash = (hash << 16) ^ tmp;
        data += 4;
        hash += hash >> 11;
    }

    switch (rem) {
    case 3: hash += *(const uint16_t*)data;
            hash ^= hash << 16;
            hash ^= ((signed char)data[2]) << 18;
            hash += hash >> 11;
            break;
    case 2: hash += *(const uint16_t*)data;
            hash ^= hash << 11;
            hash += hash >> 17;
            break;
    case 1: hash += (signed char)*data;
            hash ^= hash << 10;
            hash += hash >> 1;
            break;
    }

    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}

struct FlowKey {
	uint8_t src_ip[16];
	uint8_t dst_ip[16];
	uint16_t src_port;
	uint16_t dst_port;
	uint16_t vlan_id;
	uint8_t proto;
	uint8_t ip_version;
};

template<typename Int>
__forceinline__  __device__ static FlowKey
create_direct_key( const Int* __restrict__ src_ip,  const Int*  __restrict__ dst_ip,
	uint16_t src_port, uint16_t dst_port, uint8_t proto, IP ip_version, uint16_t vlan_id) noexcept
{
	FlowKey res;
	if (ip_version == IP::v4) {   
		*reinterpret_cast<uint64_t*>(&res.src_ip[0]) = 0;
		*reinterpret_cast<uint32_t*>(&res.src_ip[8]) = 0x0000FFFF;
		*reinterpret_cast<uint32_t*>(&res.src_ip[12]) = *reinterpret_cast<const uint32_t*>(src_ip);
		*reinterpret_cast<uint64_t*>(&res.dst_ip[0]) = 0;
		*reinterpret_cast<uint32_t*>(&res.dst_ip[8]) = 0x0000FFFF;
		*reinterpret_cast<uint32_t*>(&res.dst_ip[12]) = *reinterpret_cast<const uint32_t*>(dst_ip);
	} else if (ip_version == IP::v6) {
		memcpy(&res.src_ip, src_ip, 16);
		memcpy(&res.dst_ip, dst_ip, 16);
	}
	res.src_port = src_port;
	res.dst_port = dst_port;
	res.proto = proto;
	res.ip_version = ip_version;
	res.vlan_id = vlan_id;
	return res;
}

template<typename Int>
__forceinline__  __device__ static FlowKey
create_reversed_key(const  Int* __restrict__  src_ip, const Int*   __restrict__ dst_ip,
	uint16_t src_port, uint16_t dst_port, uint8_t proto, IP ip_version, uint16_t vlan_id) noexcept
{
	FlowKey res;
	if (ip_version == IP::v4) {   
		*reinterpret_cast<uint64_t*>(&res.dst_ip[0]) = 0;
		*reinterpret_cast<uint32_t*>(&res.dst_ip[8]) = 0x0000FFFF;
		*reinterpret_cast<uint32_t*>(&res.dst_ip[12]) = *reinterpret_cast<const uint32_t*>(src_ip);
		*reinterpret_cast<uint64_t*>(&res.src_ip[0]) = 0;
		*reinterpret_cast<uint32_t*>(&res.src_ip[8]) = 0x0000FFFF;
		*reinterpret_cast<uint32_t*>(&res.src_ip[12]) = *reinterpret_cast<const uint32_t*>(dst_ip);
	} else if (ip_version == IP::v6) {
		memcpy(&res.src_ip, dst_ip, 16);
		memcpy(&res.dst_ip, src_ip, 16);
	}
	res.src_port = dst_port;
	res.dst_port = src_port;
	res.proto = proto;
	res.ip_version = ip_version;
	res.vlan_id = vlan_id;
	return res;
}

Packet* packets_dev = nullptr;

__global__ void hash( Packet* __restrict__ packets_dev, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	Packet& packet = packets_dev[idx];
	FlowKey direct_key = create_direct_key(
		&packet.src_ip, &packet.dst_ip,
		packet.src_port, packet.dst_port,
		packet.ip_proto, (IP)packet.ip_version, packet.vlan_id);
	FlowKey reverse_key = create_reversed_key(
		&packet.src_ip, &packet.dst_ip,
		packet.src_port, packet.dst_port,
		packet.ip_proto, (IP)packet.ip_version, packet.vlan_id);

	packet.direct_hash = super_fast_hash((const char*)&direct_key, sizeof(FlowKey));
	packet.reverse_hash = super_fast_hash((const char*)&reverse_key, sizeof(FlowKey));
}

void hash_burst_gpu(PacketBlock& parsed_packets)
{
	int threadsPerBlock = 512;
	int numBlocks = (parsed_packets.cnt + threadsPerBlock - 1) / threadsPerBlock;
	if (parsed_packets.cnt != 0) {
		hash<<<numBlocks, threadsPerBlock>>>(packets_dev, parsed_packets.cnt);
    	cudaDeviceSynchronize();  
	}
	/*std::for_each_n(buffer, buffer_size, [&, index = 0](const Packet& packet) mutable {
		packet.direct_hash = packet.reverse_hash = 0;
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
	});*/

	//int blockSize = 4;
	//int numBlocks = (buffer_size + blockSize - 1) / buffer_size;
    
}

void gpu_haher_init(Packet* packets)
{
	cudaHostGetDevicePointer((void**)&packets_dev, (void*)packets, 0);
	//cudaMalloc((void **)&hashes, sizeof(FlowHash) * 100);
	//cudaMalloc((void **)&keys, sizeof(Key) * 100);
}

void gpu_haher_close()
{
	//cudaFree(hashes);
	//cudaFree(keys);
}

} // namespace ipxp
