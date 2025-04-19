
#include "cudaPacketBlock.cuh"

namespace ipxp {


PacketBlock* getCudaPacketBlock(size_t pkts_size)
{
    //Packet* packets = nullptr;
    //cudaHostAlloc(&packets, sizeof(Packet) * pkts_size, cudaHostAllocMapped);
    auto res = new PacketBlock(pkts_size);
    for (size_t i = 0; i < pkts_size; ++i) {
        cudaHostAlloc((void**)&res->pkts[i].packet, 256, cudaHostAllocMapped);
        cudaHostGetDevicePointer((void**)&res->pkts[i].packet_dev, (void*)res->pkts[i].packet, 0);
    }
    return res;
}

void freeCudaPacketBlock(PacketBlock* block)
{
    cudaFreeHost(block->pkts);
    block->pkts = nullptr;
}

} // namespace ipxp