//
// Created by zaida on 26.01.2024.
//

#ifndef IPFIXPROBE_CACHE_GAFLOWCACHE_HPP
#define IPFIXPROBE_CACHE_GAFLOWCACHE_HPP

#include "cache.hpp"
#include "gacacheoptparser.hpp"
#include "gaconfiguration.hpp"

namespace ipxp {
// Trida rozsiruje NHTFlowCache tak, aby se pouzivala predem nastavena configurace k presunu flow v radku
// namisto standartniho LRU
class GAFlowCache : public NHTFlowCache{
public:
    //GAFlowCache();
    ~GAFlowCache() override = default;
    void init(OptionsParser& parser) override;
    OptionsParser* get_parser() const override;
    void set_configuration(const GAConfiguration& src) noexcept;
    GAConfiguration get_configuration() const noexcept;
    void print_report() const noexcept override;
protected:
    void get_opts_from_parser(const GACacheOptParser& parser);
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept override;

    std::string m_infilename = ""; ///< Name of file that contains configuration
    GAConfiguration m_configuration; ///< Configuration that will be used to move flows in cache line
private:
    std::vector<uint32_t> m_unpacked_configuration; ///< Decoded m_configuration for easier work with it.
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GAFLOWCACHE_HPP
