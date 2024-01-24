//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_FLOWRECORD_HPP
#define IPFIXPROBE_CACHE_FLOWRECORD_HPP

namespace ipxp {

class FlowRecord {
    uint64_t m_hash;
public:
    Flow m_flow;

    FlowRecord();
    ~FlowRecord();

    void erase();
    void reuse();

    inline bool is_empty() const;
    inline bool belongs(uint64_t pkt_hash) const;
    void create(const Packet& pkt, uint64_t pkt_hash);
    void update(const Packet& pkt, bool src);
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_FLOWRECORD_HPP
