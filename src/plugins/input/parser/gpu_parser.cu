#include "gpu_parser.cuh"
#include "parser.cu"

namespace ipxp{

/*__device__ void parse_packet(
	Packet* opt,
	ParserStats& stats,
	struct timeval ts,
	const uint8_t* data,
	uint16_t len,
	uint16_t caplen);
*/
ParserStats* stats;
extern Packet* packets_dev;
ParserStats* stats_dev = nullptr;	

__global__ void test(){}

__global__ void parse(Packet* packets, size_t size, ParserStats* stats_dev) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	parse_packet(packets + threadIdx.x, *stats_dev);
}

//void parse_burst_gpu(PacketBlock& parsed_result, const std::vector<PacketData>& packets)
void parse_burst_gpu(PacketBlock& parsed_result)
{
	//int blockSize = 4;
	//int numBlocks = (parsed_result.cnt + blockSize - 1) / blockSize;
	
	
	//test<<<1, 1>>>();
	/*for (int i = 0; i < parsed_result.cnt; ++i) {
		cudaHostGetDevicePointer((void**) &parsed_result.pkts[i].packet_dev, (void*)parsed_result.pkts[i].packet, 0);
	}*/
	int threadsPerBlock = 512;
	int numBlocks = (parsed_result.cnt + threadsPerBlock - 1) / threadsPerBlock;
	if (parsed_result.cnt != 0) {
		parse<<<numBlocks, threadsPerBlock>>>(packets_dev, parsed_result.cnt, stats_dev);
    	cudaDeviceSynchronize();  
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

void init_gpu_parser(Packet* packets) {
	cudaHostAlloc(&stats, sizeof(*stats), cudaHostAllocMapped);
	//cudaHostGetDevicePointer((void**)&packets_dev, (void*)packets, 0);
	cudaHostGetDevicePointer((void**)&stats_dev, (void*)stats, 0);

	return;
	/*cudaMalloc((void **)&parsed_packets, sizeof(Packet) * 100);
	cudaHostAlloc(&packets_data, sizeof(PacketData) * 100, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&packets_data_dev, packets_data, 0);
	for (int i = 0; i < 100; ++i) {
		cudaHostAlloc((void**)&packets_data[i].data, 256, cudaHostAllocMapped);
		cudaHostGetDevicePointer((void**)&packets_data_dev[i].data, (void*)packets_data[i].data, 0);
	}*/
}

void close_gpu_parser() {
	return;
	/*cudaFree(parsed_packets);
	std::for_each(packets_data, packets_data + 100, [](PacketData& data){
		cudaFree((void*)(data.data));
	});
	cudaFree(packets_data);*/
}

ParserStats get_stats() {
	ParserStats result;
	cudaMemcpy(&result, &stats, sizeof(ParserStats), cudaMemcpyDeviceToHost);
	return result;
}

} // namespace ipxp