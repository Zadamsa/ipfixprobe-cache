#include "gpu_parser.cuh"
#include "parser.cuh"

namespace ipxp{

__device__ void parse_packet(
	parser_opt_t* opt,
	ParserStats& stats,
	struct timeval ts,
	const uint8_t* data,
	uint16_t len,
	uint16_t caplen);

__global__ void parse(struct ndp_packet* data, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	parser_opt_t opt = {nullptr, false, false, 0};
	ParserStats stats;
	if (idx < size) {
		parse_packet(&opt, stats, opt.pblock->pkts[idx].ts, data[idx].data, data[idx].data_length, data[idx].data_length);
	}
}

 void parse_burst_gpu(
	parser_opt_t* opt,
	ParserStats* stats,
	struct ndp_packet* buffer,
	size_t buffer_size)
{
	int blockSize = 4;
	int numBlocks = (buffer_size + blockSize - 1) / blockSize;
	// allocate memory on the GPU
	parse<<<numBlocks, blockSize>>>(buffer, buffer_size);

}

extern "C" void init_gpu_parser() {

}

extern "C" void close_gpu_parser() {

}

} // namespace ipxp