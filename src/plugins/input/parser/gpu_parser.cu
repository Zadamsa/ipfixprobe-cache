#include "gpu_parser.cuh"
#include "parser.cuh"

namespace ipxp{

Packet* parsed_packets = nullptr;
__device__ ParserStats stats;

PacketData* packets_data;
timeval* timestamps;

__device__ void parse_packet(
	Packet* opt,
	ParserStats& stats,
	struct timeval ts,
	const uint8_t* data,
	uint16_t len,
	uint16_t caplen);

__global__ void parse(Packet* parsed_packets, PacketData* packets_data, timeval* timestamps, size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	parse_packet(&parsed_packets[idx], stats, timestamps[idx], packets_data[idx].data, packets_data[idx].length, packets_data[idx].length);
}

void parse_burst_gpu(PacketBlock& parsed_result, const std::vector<PacketData>& packets)
{
	int blockSize = 4;
	int numBlocks = (packets.size() + blockSize - 1) / blockSize;
	//cudaMemcpy(raw_packets, packets.data(), packets.size() * sizeof(packets[0]), cudaMemcpyHostToDevice);
	std::for_each(packets.begin(), packets.end(), [index = 0](const PacketData& packet) mutable{
		packets_data[index].length = std::min<size_t>(packet.length, 256);
		cudaMemcpy((void*)(packets_data[index].data), packet.data, packets_data[index].length, cudaMemcpyHostToDevice);
		packets_data[index].ts = packet.ts;
		index++;
	});
	parse<<<numBlocks, blockSize>>>(parsed_packets, packets_data, timestamps, packets.size());
	cudaMemcpy(&parsed_result, parsed_packets, packets.size() * sizeof(Packet), cudaMemcpyDeviceToHost);
	parsed_result.cnt = packets.size();
}

void init_gpu_parser() {
	cudaMalloc((void **)&parsed_packets, sizeof(Packet) * 100);
	cudaMalloc((void **)&packets_data, sizeof(PacketData) * 100);
	std::for_each(packets_data, packets_data + 100, [](PacketData& data){
		cudaMalloc(&data.data, 256);
	});
}

void close_gpu_parser() {
	cudaFree(parsed_packets);
	std::for_each(packets_data, packets_data + 100, [](PacketData& data){
		cudaFree((void*)(data.data));
	});
	cudaFree(packets_data);
}

ParserStats get_stats() {
	ParserStats result;
	cudaMemcpy(&result, &stats, sizeof(ParserStats), cudaMemcpyDeviceToHost);
	return result;
}

} // namespace ipxp