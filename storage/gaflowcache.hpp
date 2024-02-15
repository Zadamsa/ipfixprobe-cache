//
// Created by zaida on 26.01.2024.
//

#ifndef IPFIXPROBE_CACHE_GAFLOWCACHE_HPP
#define IPFIXPROBE_CACHE_GAFLOWCACHE_HPP

#include "cache.hpp"
#include "gacacheoptparser.hpp"
#include "gaconfiguration.hpp"

namespace ipxp {

class GAFlowCache : public NHTFlowCache{
public:
    GAFlowCache();
    ~GAFlowCache() override = default;
    void init(OptionsParser& parser) override;
    OptionsParser* get_parser() const override;
    void set_configuration(const GAConfiguration& src) noexcept;
    GAConfiguration get_configuration() const noexcept;
    void print_report() const noexcept override;
protected:
    void get_opts_from_parser(const GACacheOptParser& parser);
    uint32_t enhance_existing_flow_record(uint32_t flow_index) noexcept override;
    //uint32_t make_place_for_record(uint32_t line_index) noexcept override;
private:
    std::string m_infilename = "";
    GAConfiguration m_configuration;
    std::vector<uint32_t> m_unpacked_configuration;
    //uint32_t m_insert_pos;
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GAFLOWCACHE_HPP
