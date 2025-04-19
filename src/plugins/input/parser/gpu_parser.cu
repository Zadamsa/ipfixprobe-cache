#include "gpu_parser.cuh"
#include "parser.cu"

namespace ipxp{

Packet* parsed_packets = nullptr;

PacketData* packets_data;
PacketData* packets_data_dev;
timeval* timestamps;

/*__device__ void parse_packet(
	Packet* opt,
	ParserStats& stats,
	struct timeval ts,
	const uint8_t* data,
	uint16_t len,
	uint16_t caplen);
*/
ParserStats* stats;

__global__ void test(){}

__global__ void parse(Packet* packets, size_t size, ParserStats* stats_dev) {
	//int idx = threadIdx.x;
	if (threadIdx.x >= size) {
		return;
	}
	parse_packet(packets + threadIdx.x, *stats_dev);
}

//void parse_burst_gpu(PacketBlock& parsed_result, const std::vector<PacketData>& packets)
void parse_burst_gpu(PacketBlock& parsed_result)
{
	//int blockSize = 4;
	//int numBlocks = (parsed_result.cnt + blockSize - 1) / blockSize;
	Packet* packets_dev = nullptr;
	ParserStats* stats_dev = nullptr;	
	cudaHostGetDevicePointer((void**)&packets_dev, (void*)parsed_result.pkts, 0);
	cudaHostGetDevicePointer((void**)&stats_dev, (void*)stats, 0);
	//test<<<1, 1>>>();
	for (int i = 0; i < parsed_result.cnt; ++i) {
		cudaHostGetDevicePointer((void**) &parsed_result.pkts[i].packet_dev, (void*)parsed_result.pkts[i].packet, 0);
	}
	parse<<<1, 64>>>(packets_dev, parsed_result.cnt, stats_dev);
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();  
	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
	//cudaMemcpy(raw_packets, packets.data(), packets.size() * sizeof(packets[0]), cudaMemcpyHostToDevice);
	/*std::for_each(packets.begin(), packets.end(), [index = 0](const PacketData& packet) mutable{
		packets_data[index].length = std::min<size_t>(packet.length, 256);
		cudaMemcpy((void*)(packets_data[index].data), packet.data, packets_data[index].length, cudaMemcpyHostToDevice);
		packets_data[index].ts = packet.ts;
		index++;
	});
	parse<<<numBlocks, blockSize>>>(parsed_packets, packets_data, timestamps, packets.size());
	cudaMemcpy(&parsed_result, parsed_packets, packets.size() * sizeof(Packet), cudaMemcpyDeviceToHost);
	parsed_result.cnt = packets.size();*/
}

void init_gpu_parser() {
	cudaHostAlloc(&stats, sizeof(*stats), cudaHostAllocMapped);
	cudaMalloc((void **)&parsed_packets, sizeof(Packet) * 100);
	cudaHostAlloc(&packets_data, sizeof(PacketData) * 100, cudaHostAllocMapped);
	//cudaMalloc((void **)&packets_data, sizeof(PacketData) * 100);
	/*std::for_each(packets_data, packets_data + 100, [](PacketData& data){
		cudaMalloc(&data.data, 256);
	});*/
	cudaHostGetDevicePointer(&packets_data_dev, packets_data, 0);
	for (int i = 0; i < 100; ++i) {
		cudaHostAlloc((void**)&packets_data[i].data, 256, cudaHostAllocMapped);
		cudaHostGetDevicePointer((void**)&packets_data_dev[i].data, (void*)packets_data[i].data, 0);
	}
}

void close_gpu_parser() {
	return;
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