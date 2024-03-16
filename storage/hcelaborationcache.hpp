#ifndef CACHE_CPP_HCFLOWCACHE_HPP
#define CACHE_CPP_HCFLOWCACHE_HPP
#include "gaelaborationcache.hpp"
namespace ipxp {

class HCElaborationFlowCache : public GAElaborationCache{
protected:
    void start_workers() override;
    void save_best_configuration(bool parent_exists,const CacheStatistics& parent_statics) const override;
    void read_taboo_list();
    void save_taboo_list() const;
    std::vector<GAConfiguration> m_taboo_list;
    double m_heat = 0;
};

} // namespace ipxp

#endif // CACHE_CPP_HCFLOWCACHE_HPP
