#include "hcelaborationcache.hpp"
#include <unistd.h>
namespace ipxp {

__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("hcelabcache", []() { return new HCElaborationFlowCache(); });
    register_plugin(&rec);
}

std::string HCElaborationFlowCache::get_name() const noexcept{
    return "hcelabcache";
}


void HCElaborationFlowCache::read_taboo_list(){
    auto filename = m_infilename + ".taboo";
    std::ifstream ifs(filename,std::ios::binary);
    if (!ifs)
        throw PluginError("Can't open HC taboo file: " + filename);
    uint16_t taboo_size;
    ifs.read((char*)&taboo_size,sizeof(taboo_size));
    for(decltype(taboo_size) i = 0; i < taboo_size; i++ ){
        GAConfiguration config(m_line_size);
        config.read_from_file(ifs);
        m_taboo_list.emplace_back(std::move(config));
    }
    ifs.read((char*)&m_heat,sizeof(m_heat));
    if (!ifs)
        throw PluginError("Invalid HC taboo file: " + filename);
}
void HCElaborationFlowCache::save_taboo_list(){
    auto filename = m_outfilename + ".taboo";
    std::ofstream ofs(filename,std::ios::binary);
    if (!ofs)
        throw PluginError("Can't open HC taboo savefile: " + filename);

    if (m_taboo_list.size() > m_generation_size)
        m_taboo_list.erase(m_taboo_list.begin());
    uint16_t taboo_size = m_taboo_list.size();
    ofs.write((char*)&taboo_size,sizeof(taboo_size));
    for(const auto& config : m_taboo_list )
        config.write_to_file(ofs);
    ofs.write((char*)&m_heat,sizeof(m_heat));
    if (!ofs)
        throw PluginError("Can't save HC taboo file to  " + filename);
}

void HCElaborationFlowCache::start_workers(GAElaborationCacheOptParser* parser){
    if (m_infilename != "")
        read_taboo_list();
    GAElaborationCache::start_workers(parser);
}

void HCElaborationFlowCache::create_generation(std::vector<GAConfiguration>& configurations, const GAConfiguration& default_config) const noexcept
{
    // Vytvorit mutovane konfigurace + kontrola ze se nevygenrovali stejne konfigurace
    for (uint32_t i = 0; i < m_generation_size; i++) {
        bool uniq = false;
        configurations.emplace_back(default_config);
        while (!uniq) {
            do {
                configurations[i] = configurations[i].mutate();
            } while (in_taboo_list(configurations[i]));
            uniq = true;
            for (int32_t k = i - 1; i >= 1 && k >= 0 && uniq; k--)
                uniq = configurations[i] != configurations[k];
        }
    }
}

bool HCElaborationFlowCache::in_taboo_list(const GAConfiguration& config) const noexcept{
    for(const auto& taboo_config: m_taboo_list)
        if (GAConfiguration::distance(taboo_config,config) < m_heat * -0.5 + 20)
            return true;
    return false;
}

void HCElaborationFlowCache::save_best_configuration(bool parent_exists,const CacheStatistics& parent_statics){
    auto best_config = m_caches[0]->get_configuration();
    auto best_stats = m_caches[0]->get_total_statistics();
    // Pokud rodicovska konfigurace existuje, a je lepsi nez libovolny z potomku, ulozit ji
    if (parent_exists && parent_statics < best_stats){
        best_config = m_configuration;
        best_stats = parent_statics;
        std::mt19937 generator = std::mt19937(std::random_device()());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for(auto it = m_caches.begin(); it != m_caches.end(); ++it){
            const auto config = (*it)->get_configuration();
            const auto denom =((m_heat + (it - m_caches.begin())) * (-3.0/40) - 2);
            auto exp = std::exp((double)GAConfiguration::distance(config,best_config)/(6*denom));
            if (exp > distribution(generator)){
                std::cout << std::to_string(it - m_caches.begin()) + "-th configuration replaced\n";
                best_config = config;
                best_stats = (*it)->get_total_statistics();
            }
        }
    }

    bool global_min_exists = false;
    CacheStatistics global_min_statics;
    for(int i = 0; i < 5 && !global_min_exists; i++) {
        global_min_exists = true;
        try {
            global_min_statics.read_from_file(m_outfilename + ".global_min.stats");
        } catch (...) {
            global_min_exists = false;
            usleep(100);
        }
    }
    if (!global_min_exists || m_infilename == "" || best_stats < global_min_statics){
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
        std::cout<< "Just has replaced "<<global_min_statics.m_not_empty << " with new minimum " << best_stats.m_not_empty << "\n";
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
        best_config.write_to_file(m_outfilename + ".global_min");
        best_stats.write_to_file(m_outfilename + ".global_min.stats");
    }
    m_taboo_list.emplace_back(best_config);
    m_heat++;
    save_taboo_list();



    best_config.write_to_file(m_outfilename);
    best_stats.write_to_file(m_outfilename + ".stats");
}

} // namespace ipxp