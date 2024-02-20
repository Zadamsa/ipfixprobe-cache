//
// Created by zaida on 26.01.2024.
//

#ifndef IPFIXPROBE_CACHE_GAELABORATIONCACHE_HPP
#define IPFIXPROBE_CACHE_GAELABORATIONCACHE_HPP

#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include <cstdint>
#include <vector>
#include <ipfixprobe/storage.hpp>
#include <ipfixprobe/flowifc.hpp>
#include "gaelaborationcacheoptparser.hpp"
#include "gaflowcache.hpp"

namespace ipxp {
// Trida urcena k vyvoji novych konfiguraci, ktere pak pouzije GAFlowCache
// Navod na vyvoj a pouziti vygenerovane spravy:
// ./ipfixprobe -i 'pcap;file=/mnt/e/zainudam/downloads/traffic.pcap' -s 'gaelabcache;outputconfig=gaconfig_traffic.bin;g=16' -- vygeneruje g=16 nahodnych sprav a ohodnoti.
// Nejlepsi sprava bude ulozena do gaconfig_traffic.bin, statistiky do gaconfig_traffic.bin.stats.
// for i in `seq 1 10`; do ./ipfixprobe -i 'pcap;file=/mnt/e/zainudam/downloads/traffic.pcap' -s 'gaelabcache;outputconfig=gaconfig_traffic.bin;ic=gaconfig_traffic.bin;g=16;'; done --
// nacte gaconfig_traffic.bin a gaconfig_traffic.bin.stats, vygeneruje g=16 mutaci, vyhodnoti, ulozi nejlepsi spravu do gaconfig_traffic.bin a gaconfig_traffic.bin.stats. To cele se opakuje ve
// smycce, tady se vyhodnoti 10 generaci.
// ./ipfixprobe -i 'pcap;file=/mnt/e/zainudam/downloads/traffic.pcap' -s 'gacache;ic=gaconfig_traffic.bin;' -- GAFlowCache ohodnoti spravu gaconfig_traffic.bin.
class GAElaborationCache : public GAFlowCache {
public:
    void init(OptionsParser& parser) override;
    void set_queue(ipx_ring_t* queue) override;
    OptionsParser* get_parser() const override;
    std::string get_name() const noexcept;
    int put_pkt(Packet& pkt) override;
    void finish() override;
private:
    void cache_worker(uint32_t worker_id) noexcept;
    void get_opts_from_parser(const GAElaborationCacheOptParser& parser);
    void create_generation(std::vector<GAConfiguration>& configurations, const GAConfiguration& default_config) const noexcept;

    uint8_t m_generation_size = 0;
    std::vector<std::thread> m_threads;
    std::vector<std::unique_ptr<GAFlowCache>> m_caches;
    std::string m_outfilename = "";
    Packet* m_pkt_ptr = nullptr;
    bool m_exit = false;
    std::mutex m_mutex;
    std::condition_variable m_new_pkt_cond;
    std::condition_variable m_done_cond;
    std::vector<bool> m_done;
    std::atomic<int> m_pkt_id = 0;
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GAELABORATIONCACHE_HPP
