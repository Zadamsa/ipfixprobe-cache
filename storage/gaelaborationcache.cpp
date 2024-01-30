//
// Created by zaida on 26.01.2024.
//

#include "gaelaborationcache.hpp"
#include "gaconfiguration.hpp"

namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("gaelabcache", []() { return new GAElaborationCache(); });
    register_plugin(&rec);
}

GAElaborationCache::GAElaborationCache():GAFlowCache(){}
GAElaborationCache::~GAElaborationCache(){
    m_exit = true;
    m_done = std::vector<bool>(m_generation_size,false);
    m_done_cond.notify_all();

    for(auto& thr : m_threads)
        thr.join();
    const auto best_example = std::min_element(m_caches.begin(), m_caches.end(),
                     [](const auto& a, const auto& b) {
                         return a->get_total_statistics().m_not_empty < b->get_total_statistics().m_not_empty;
                     }
    );
    if (best_example != m_caches.end())
        (*best_example)->get_configuration().write_to_file(m_outfilename);
}

void GAElaborationCache::get_opts_from_parser(const GAElaborationCacheOptParser& parser){
    GAFlowCache::get_opts_from_parser(parser);
    m_generation_size = parser.m_generation_size;
    m_infilename = parser.m_infilename;
    m_outfilename = parser.m_outfilename;
}

void GAElaborationCache::init(OptionsParser& in_parser) {
    auto parser = dynamic_cast<GAElaborationCacheOptParser*>(&in_parser);
    if (!parser)
        throw PluginError("Bad options parser for GAElaborationCache");
    get_opts_from_parser(*parser);
    parser->m_infilename = "";
    std::vector<GAConfiguration> configurations;
    GAConfiguration x(m_line_size);
    GAConfiguration y(x);
    if (m_infilename != ""){
        GAConfiguration default_config(m_line_size);
        default_config.read_from_file(m_infilename);
        create_generation(configurations, default_config);
    }else{
        for (uint32_t i = 0; i < m_generation_size; i++)
            configurations.emplace_back(m_line_size);
    }

    m_done = std::vector<bool>(m_generation_size,true);
    for(uint32_t i = 0; i < m_generation_size; i++){
        m_caches.emplace_back(std::make_unique<GAFlowCache>());
        m_caches.back()->set_queue(m_export_queue);
        m_caches.back()->set_configuration(configurations[i]);
        m_caches.back()->init(*parser);
        m_threads.emplace_back(&GAElaborationCache::cache_worker,this,i);
    }
}

void GAElaborationCache::create_generation(std::vector<GAConfiguration>& configurations, const GAConfiguration& default_config) const noexcept{
    for(uint32_t i = 0; i < m_generation_size; i++){
        bool uniq = false;
        configurations.emplace_back(default_config);
        while(!uniq) {
            configurations[i] = configurations[i].mutate();
            uniq = true;
            for (int32_t k = i - 1; i >= 1 && k >= 0 && uniq; k--)
                uniq = configurations[i] != configurations[k];
        }
    }
}

OptionsParser* GAElaborationCache::get_parser() const{
    return new GAElaborationCacheOptParser();
}
std::string GAElaborationCache::get_name() const noexcept{
    return "gaelabcache";
}
void GAElaborationCache::set_queue(ipx_ring_t* queue){
    GAFlowCache::set_queue(queue);
}
int GAElaborationCache::put_pkt(Packet& pkt){
    m_statistics.m_hits++;
    std::unique_lock ul(m_mutex);
    m_pkt_ptr = &pkt;
    m_done = std::vector<bool>(m_generation_size,false);
    m_done_cond.notify_all();
    m_new_pkt_cond.wait(ul,[this](){return std::all_of(m_done.begin(), m_done.end(), [](bool d) {
                                           return d;
                                       });});
    //m_finished_count = 0;
    m_pkt_ptr = nullptr;
    return 0;
}

void GAElaborationCache::cache_worker(uint32_t worker_id) noexcept{
    while(!m_exit) {
        std::unique_lock ul(m_mutex);
        m_done_cond.wait(ul, [this,worker_id]() { return !m_done[worker_id]; });
        if (m_exit)
            return ;
        ul.unlock();
        m_caches[worker_id]->put_pkt(*m_pkt_ptr);
        ul.lock();
        m_done[worker_id] = true;
        m_new_pkt_cond.notify_one();
    }
}
void GAElaborationCache::finish(){}

} // namespace ipxp