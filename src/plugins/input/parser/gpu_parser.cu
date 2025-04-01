#include "parser.hpp"

#include <ipfixprobe/parser-stats.hpp>

__global__ void processElements(struct ndp_packet* data, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	parser_opt_t opt = {nullptr, false, false, 0};
	ParserStats stats;
	if (idx < size) {
		parse_packet(&opt, stats, data[idx]->ts, data[idx]->data, data[idx]->data_length, data[idx]->data_length);
	}
}


extern "C" void parse_burst_gpu(
	parser_opt_t* opt,
	ParserStats& stats,
	struct ndp_packet* buffer,
	size_t buffer_size)
{
	int blockSize = 4;
	int numBlocks = (buffer_size + blockSize - 1) / blockSize;
	// allocate memory on the GPU
	parse<<<numBlocks, blockSize>>>(buffer, buffer_size);

}