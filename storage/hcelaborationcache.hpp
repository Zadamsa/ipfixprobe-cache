#ifndef CACHE_CPP_HCFLOWCACHE_HPP
#define CACHE_CPP_HCFLOWCACHE_HPP
#include "gaelaborationcache.hpp"
namespace ipxp {

class HCElaborationFlowCache : public GAElaborationCache{
public:
    std::string get_name() const noexcept override;
protected:
    void start_workers(GAElaborationCacheOptParser* parser) override;
    void save_best_configuration(bool parent_exists,const CacheStatistics& parent_statics) override;
    void read_taboo_list();
    void save_taboo_list();
    void create_generation(std::vector<GAConfiguration>& configurations, const GAConfiguration& default_config) const noexcept;
    bool in_taboo_list(const GAConfiguration& config) const noexcept;
    std::vector<GAConfiguration> m_taboo_list;
    double m_heat = 0;
};

} // namespace ipxp

#endif // CACHE_CPP_HCFLOWCACHE_HPP
