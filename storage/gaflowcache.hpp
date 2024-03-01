//
// Created by zaida on 26.01.2024.
//

#ifndef IPFIXPROBE_CACHE_GAFLOWCACHE_HPP
#define IPFIXPROBE_CACHE_GAFLOWCACHE_HPP

#include "cache.hpp"
#include "gacacheoptparser.hpp"
#include "gaconfiguration.hpp"
#include "packetclassifier.hpp"

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
    //uint32_t free_place_in_full_line(uint32_t line_begin) noexcept override;
    int insert_pkt(Packet& pkt) noexcept override;
    std::string m_infilename = ""; ///< Name of file that contains configuration
    GAConfiguration m_configuration; ///< Configuration that will be used to move flows in cache line
    bool is_being_flooded(const Packet& Pkt) noexcept override;
    void export_graph_data(const Packet& pkt) override;

private:
    std::vector<uint32_t> m_unpacked_configuration; ///< Decoded m_configuration for easier work with it.
    uint32_t m_short_pos = 0;
    uint32_t m_medium_pos = 0;
    uint32_t m_long_pos = 0;
    uint32_t m_never_pos = 0;
    PacketDistance m_pkt_dist;
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GAFLOWCACHE_HPP
