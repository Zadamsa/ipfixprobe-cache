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

class GAElaborationCache : public GAFlowCache {
public:
    GAElaborationCache();
    ~GAElaborationCache() override;
    //void init(const char* params) override;
    void init(OptionsParser& parser) override;
    void set_queue(ipx_ring_t* queue) override;
    OptionsParser* get_parser() const override;
    std::string get_name() const noexcept;
    int put_pkt(Packet& pkt) override;
private:
    void cache_worker(uint32_t worker_id) noexcept;
    void get_opts_from_parser(const GAElaborationCacheOptParser& parser);
    void create_generation(std::vector<GAConfiguration>& configurations, const GAConfiguration& default_config) const noexcept;
    void finish() override;

    uint8_t m_generation_size = 0;
    std::atomic<uint8_t> m_finished_count = 0;
    std::vector<std::thread> m_threads;
    std::vector<std::unique_ptr<GAFlowCache>> m_caches;
    std::string m_infilename = "";
    std::string m_outfilename = "";
    Packet* m_pkt_ptr = nullptr;
    bool m_exit = false;
    std::mutex m_mutex;
    std::condition_variable m_new_pkt_cond;
    std::condition_variable m_done_cond;
    std::vector<bool> m_done;
    std::atomic<int> m_started = 0;
    std::atomic<int> m_pkt_id = 0;
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GAELABORATIONCACHE_HPP
