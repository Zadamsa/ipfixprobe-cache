#include "gpu_parser.cuh"
#include "parser.cu"

namespace ipxp{

ParserStats* stats;
extern Packet* packets_dev;
ParserStats* stats_dev = nullptr;	

__global__ void test(){}

__global__ void parse(Packet* packets, size_t size, ParserStats* stats_dev) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}
	parse_packet(packets + idx, *stats_dev);
}

void parse_burst_gpu(PacketBlock& parsed_result)
{
	int threadsPerBlock = 1024;
	int numBlocks = (parsed_result.cnt + threadsPerBlock - 1) / threadsPerBlock;
	if (parsed_result.cnt != 0) {
		parse<<<numBlocks, threadsPerBlock>>>(packets_dev, parsed_result.cnt, stats_dev);
    	cudaDeviceSynchronize();  
	}
}

void init_gpu_parser(Packet* packets) {
	cudaHostAlloc(&stats, sizeof(*stats), cudaHostAllocMapped);
	cudaHostGetDevicePointer((void**)&stats_dev, (void*)stats, 0);

	return;
}

ParserStats get_stats() {
	ParserStats result;
	cudaMemcpy(&result, &stats, sizeof(ParserStats), cudaMemcpyDeviceToHost);
	return result;
}

} // namespace ipxp