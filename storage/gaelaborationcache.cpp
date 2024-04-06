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

OptionsParser* GAElaborationCache::get_parser() const{
    return new GAElaborationCacheOptParser();
}

std::string GAElaborationCache::get_name() const noexcept{
    return "gaelabcache";
}

void GAElaborationCache::get_opts_from_parser(const GAElaborationCacheOptParser& parser){
    GAFlowCache::get_opts_from_parser(parser);
    m_generation_size = parser.m_generation_size;
    m_infilename = parser.m_infilename;
    m_outfilename = parser.m_outfilename;
}

void GAElaborationCache::start_workers(GAElaborationCacheOptParser* parser){
    std::vector<GAConfiguration> configurations;
    // Pokud nebyl zadan vstupni soubor, musi GAElaborationCache vytvorit generace nahodnych jedincu
    // a ulozit konfigurace nejlepsiho.
    // Pokud vstupni soubor byl zadan, vytvori se generace mutovanych(vuci konfiguraci ze souboru) konfiguraci
    if (m_infilename != ""){
        //GAConfiguration default_config(m_line_size);
        m_configuration = GAConfiguration(m_line_size);
        m_configuration.read_from_file(m_infilename);
        create_generation(configurations, m_configuration);
    }else{
        // m_generation_size nahodnych(ale validnich) konfiguraci
        for (uint32_t i = 0; i < m_generation_size; i++)
            configurations.emplace_back(m_line_size);
    }
    //m_done = std::vector<bool>(m_generation_size,true);
    m_done = std::vector<bool>(m_generation_size,false);
    // Nastavit vygenerovane konfigurace a spustit vlakna pro kazdou cache
    for(uint32_t i = 0; i < m_generation_size; i++){
        m_caches.emplace_back(std::make_unique<GAFlowCache>());
        m_caches.back()->set_queue(m_export_queue);
        m_caches.back()->init(*parser);
        m_caches.back()->set_configuration(configurations[i]);
        m_threads.emplace_back(&GAElaborationCache::cache_worker,this,i);
    }
    //Pockat, az vsechni workery budou pripravene
    /*std::unique_lock ul(m_mutex);
    m_new_pkt_cond.wait(ul,[this](){return std::all_of(m_done.begin(), m_done.end(), [](bool d) {
                                           return d;
                                       });});*/
}

void GAElaborationCache::init(OptionsParser& in_parser) {
    auto parser = dynamic_cast<GAElaborationCacheOptParser*>(&in_parser);
    if (!parser)
        throw PluginError("Bad options parser for GAElaborationCache");
    get_opts_from_parser(*parser);
    //Nastavit prazdne jmeno souboru aby GAFlowCache se nesnazili nacist konfigurace ze souboru
    parser->m_infilename = "";
    start_workers(parser);

}

void GAElaborationCache::create_generation(std::vector<GAConfiguration>& configurations, const GAConfiguration& default_config) const noexcept{
    // Vytvorit mutovane konfigurace + kontrola ze se nevygenrovali stejne konfigurace
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


void GAElaborationCache::set_queue(ipx_ring_t* queue){
    GAFlowCache::set_queue(queue);
}
// Synchronizacni funkce pro vkladani noveho paketu do vsech cache
int GAElaborationCache::put_pkt(Packet& pkt){

    std::unique_lock ul(m_mutex);
    if (m_packets_buffer.size() == 100'000) {
        m_done_cond.notify_all();
        m_new_pkt_cond.wait(ul, [this]() {
            return std::all_of(m_done.begin(), m_done.end(), [](bool d) { return d; });
        });
        m_done = std::vector<bool>(m_generation_size,false);
        m_packets_buffer.clear();
    }
    m_packets_buffer.push_back(pkt);
    // m_pkt_ptr = &pkt;
    m_pkt_id++;

    // Pockat az vsechni cache si vlozi novy paket

    return 0;
}

// Jeden thread pro kazdou vyhodnocovanou cache, pracuje spolecne s GAElaborationCache::put_pkt()
void GAElaborationCache::cache_worker(uint32_t worker_id) noexcept{
    int last_id = 0;
    while(!m_exit){
        std::unique_lock ul(m_mutex);
        m_done_cond.wait(ul,[this,worker_id](){return (m_packets_buffer.size() == 100'000 && !m_done[worker_id]) || m_exit;});
        ul.unlock();
        if (m_done[worker_id])
            break;
        last_id += m_packets_buffer.size();
        for(auto& pkt : m_packets_buffer)
            m_caches[worker_id]->put_pkt(pkt);
        //last_id = m_pkt_id;
        m_done[worker_id] = true;
        m_new_pkt_cond.notify_one();
        //segfault pokud by se nastala desynchronizace
        /*if (m_pkt_id != last_id + 1 && !m_exit)
            *((uint16_t*)0) = 666;*/
        //if (m_exit)
        //    return;
    }
    std::cout << last_id << "packets read\n";
}

void GAElaborationCache::finish(){
    m_exit = true;
    //m_done = std::vector<bool>(m_generation_size,true);
    m_done_cond.notify_all();
    std::cout << m_pkt_id << "packets pushed back\n";
    for(auto& thr : m_threads)
        thr.join();
    for(auto& cache_ptr : m_caches)
        cache_ptr->finish();
    //Zvolit nejlepsi spravu podle statistik
    std::sort(m_caches.begin(), m_caches.end(),[](const auto& a, const auto& b) {
                                                   return a->get_total_statistics() < b->get_total_statistics();
                                                }
    );
    // Nacist statistiky rodicovske konfigurace, pokud existuje
    bool parent_exists = true;
    CacheStatistics parent_statics;
    try {
        parent_statics.read_from_file(m_infilename + ".stats");
    }catch (...){
        parent_exists = false;
    }
    save_best_configuration(parent_exists,parent_statics);
}

void GAElaborationCache::save_best_configuration(bool parent_exists,const CacheStatistics& parent_statics){
    const auto& best_example = m_caches[0];
    // Pokud rodicovska konfigurace existuje, a je lepsi nez libovolny z potomku, ulozit ji
    if (parent_exists && parent_statics < (best_example)->get_total_statistics()){
        m_configuration.write_to_file(m_outfilename);
        parent_statics.write_to_file(m_outfilename + ".stats");
    }else{
        (best_example)->get_configuration().write_to_file(m_outfilename);
        (best_example)->get_total_statistics().write_to_file(m_outfilename + ".stats");
    }
}

} // namespace ipxp