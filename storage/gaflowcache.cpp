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
    std::tie(m_short_pos,m_medium_pos,m_long_pos,m_never_pos,m_unpacked_configuration) = src.unpack();
}
OptionsParser* GAFlowCache::get_parser() const{
    return new GACacheOptParser();
}

void GAFlowCache::get_opts_from_parser(const GACacheOptParser& parser){
    NHTFlowCache::get_opts_from_parser(parser);
    m_infilename = parser.m_infilename;
}

bool GAFlowCache::is_being_flooded(const Packet& Pkt) noexcept{
    return false;
}
void GAFlowCache::export_graph_data(const Packet& pkt){}

uint32_t GAFlowCache::enhance_existing_flow_record(uint32_t flow_index) noexcept{
    uint32_t line_index = flow_index & m_line_mask;
    m_statistics.m_lookups += (flow_index - line_index + 1);
    m_statistics.m_lookups2 += (flow_index - line_index + 1) * (flow_index - line_index + 1);
    m_statistics.m_hits++;
    // Tady se pouziva rozbalena konfigurace ke zvoleni nove pozice pro flow
    cyclic_rotate_records(line_index + m_unpacked_configuration[flow_index - line_index],flow_index);
    return line_index;
}

int GAFlowCache::insert_pkt(Packet& pkt) noexcept {
    m_pkt_dist = PacketClassifier::classifyInstance(pkt.ip_proto,pkt.tcp_flags,pkt.tcp_window,pkt.ip_payload_len);
    return NHTFlowCache::insert_pkt(pkt);
}

uint32_t GAFlowCache::free_place_in_full_line(uint32_t line_begin) noexcept
{
    uint32_t line_end = line_begin + m_line_size;
    prepare_and_export(line_end - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    uint32_t flow_new_index;
    if (m_pkt_dist == PacketDistance::DISTANCE_SHORT)
        flow_new_index = line_begin + m_short_pos;
    else if (m_pkt_dist == PacketDistance::DISTANCE_MEDIUM)
        flow_new_index = line_begin + m_medium_pos;
    else if (m_pkt_dist == PacketDistance::DISTANCE_LONG)
        flow_new_index = line_begin + m_long_pos;
    else
        flow_new_index = line_begin + m_never_pos;
    cyclic_rotate_records(flow_new_index, line_end - 1);
    return flow_new_index;
}

void GAFlowCache::print_report() const noexcept{
    if (m_statistics.m_hits) {
        std::cout << "==================================================================\nTOTAL\n";
        std::cout << m_configuration.to_string() << "\n";
        std::cout << m_statistics;
    }
}

} // namespace ipxp