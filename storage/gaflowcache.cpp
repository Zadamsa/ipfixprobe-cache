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

GAFlowCache::GAFlowCache():NHTFlowCache(){};

void GAFlowCache::init(OptionsParser& in_parser){
    NHTFlowCache::init(in_parser);
    auto parser = dynamic_cast<GACacheOptParser*>(&in_parser);
    if (!parser)
        throw PluginError("Bad options parser for GAFlowCache");
    get_opts_from_parser(*parser);
    m_configuration = GAConfiguration(m_line_size);
    if (m_infilename != "")
        m_configuration.read_from_file(m_infilename);
    set_configuration(m_configuration);

}
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

    cyclic_rotate_records(line_index + m_unpacked_configuration[flow_index - line_index],flow_index);
    return line_index;
}

uint32_t GAFlowCache::make_place_for_record(uint32_t line_index) noexcept{
    uint32_t next_line = line_index + m_line_size;
    if (m_flow_table[next_line - 1]->is_empty()){
        m_statistics.m_empty++;
    }else{
        m_statistics.m_not_empty++;
        prepare_and_export(next_line - 1, FlowEndReason::FLOW_END_LACK_OF_RECOURSES);
    }
    cyclic_rotate_records(line_index + m_insert_pos,next_line - 1);
    return line_index + m_insert_pos;
}

} // namespace ipxp