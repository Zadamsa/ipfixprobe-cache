//
// Created by zaida on 26.01.2024.
//

#include "gaflowcache.hpp"

namespace ipxp {
__attribute__((constructor)) static void register_this_plugin() noexcept
{
    static PluginRecord rec = PluginRecord("gacache", []() { return new GAFlowCache(); });
    register_plugin(&rec);
}

void GAFlowCache::init(OptionsParser& in_parser){
    NHTFlowCache::init(in_parser);
    auto parser = dynamic_cast<GACacheOptParser*>(&in_parser);
    if (!parser)
        throw PluginError("Bad options parser for GAFlowCache");
    get_opts_from_parser(*parser);
    // Generuje se nahodna konfigurace
    m_configuration = GAConfiguration(m_line_size);
    // Pokud bylo zadano jmeno vstupniho souboru, pokusit se nacist konfigurace z souboru
    if (m_infilename != "")
        m_configuration.read_from_file(m_infilename);
    set_configuration(m_configuration);

}

// Vrati kopie aktualne konfiguraci
GAConfiguration GAFlowCache::get_configuration() const noexcept{
    return m_configuration;
}

void GAFlowCache::set_configuration(const GAConfiguration& src) noexcept{
    m_configuration = src;
    std::tie(m_insert_pos,m_unpacked_configuration) = src.unpack();
}
OptionsParser* GAFlowCache::get_parser() const{
    return new GACacheOptParser();
}

void GAFlowCache::get_opts_from_parser(const GACacheOptParser& parser){
    NHTFlowCache::get_opts_from_parser(parser);
    m_infilename = parser.m_infilename;
}

uint32_t GAFlowCache::enhance_existing_flow_record(uint32_t flow_index) noexcept{
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;
    // Tady se pouziva rozbalena konfigurace ke zvoleni nove pozice pro flow
    cyclic_rotate_records(line_index + m_unpacked_configuration[flow_index - line_index],flow_index);
    return line_index;
}

void GAFlowCache::print_report() const noexcept{
    if (m_statistics.m_hits) {
        std::cout << "==================================================================\nTOTAL\n";
        std::cout << m_configuration.to_string() << "\n";
        std::cout << m_statistics;
    }
}

} // namespace ipxp