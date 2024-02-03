uint32_t toeplitz_hash(const Packet& pkt, bool reversed_order) const noexcept{
    rte_thash_tuple tuple = {0};
    if (pkt.ip_version == IP::v4){
        if (!reversed_order) {
            tuple.v4.src_addr = pkt.src_ip.v4;
            tuple.v4.dst_addr = pkt.dst_ip.v4;
            tuple.v4.sport = pkt.src_port;
            tuple.v4.dport = pkt.dst_port;
        }else{
            tuple.v4.src_addr = pkt.dst_ip.v4;
            tuple.v4.dst_addr = pkt.src_ip.v4;
            tuple.v4.sport = pkt.dst_port;
            tuple.v4.dport = pkt.src_port;
        }
        return rte_softrss_be((uint32_t*)&tuple.v4, RTE_THASH_V4_L3_LEN,(uint8_t*)m_rss_key);
    }else if (pkt.ip_version == IP::v6){
        rte_ipv6_hdr hdr = {0};
        if (!reversed_order) {
            memcpy(hdr.src_addr, pkt.src_ip.v6, 16);
            memcpy(hdr.dst_addr, pkt.dst_ip.v6, 16);
            rte_thash_load_v6_addrs(&hdr, (rte_thash_tuple * ) & tuple.v6);
            tuple.v6.sport = pkt.src_port;
            tuple.v6.dport = pkt.dst_port;
        }else{
            memcpy(hdr.src_addr, pkt.dst_ip.v6, 16);
            memcpy(hdr.dst_addr, pkt.src_ip.v6, 16);
            rte_thash_load_v6_addrs(&hdr, (rte_thash_tuple * ) & tuple.v6);
            tuple.v6.sport = pkt.dst_port;
            tuple.v6.dport = pkt.src_port;
        }
        return rte_softrss_be((uint32_t*)&tuple.v6, RTE_THASH_V6_L3_LEN,(uint8_t*)m_rss_key);
    }
    return 0;


}//
// Created by zaida on 02.02.2024.
//
