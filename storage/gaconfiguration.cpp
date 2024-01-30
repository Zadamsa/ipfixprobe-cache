//
// Created by zaida on 26.01.2024.
//

#include <fstream>
#include <algorithm>
#include <string>
#include <ipfixprobe/plugin.hpp>
#include "gaconfiguration.hpp"

using namespace std::string_literals;
namespace ipxp {

GAConfiguration::GAConfiguration(uint32_t line_size): m_line_size(line_size){
    uint32_t prev = 0;
    uint32_t generated_nodes_count = 0;
    for( uint32_t i = 0; i < line_size/4; i++){
        std::uniform_int_distribution<std::mt19937::result_type> moves_dist(prev,generated_nodes_count);
        uint8_t count = m_count_dist(m_rng);
        uint8_t target = moves_dist(m_rng);
        generated_nodes_count += count;
        prev = target;
        m_moves.push_back(MoveTuple{count,target});
    }
    mutate_insert_pos(1);
    fix_counts();
    fix_targets();
}
void GAConfiguration::fix() noexcept{
    fix_counts();
    fix_targets();
}

GAConfiguration::GAConfiguration(){};

GAConfiguration GAConfiguration::mutate() const{
    GAConfiguration new_configuration = *this;
    while(new_configuration == *this) {
        new_configuration.mutate_counts(0.2);
        new_configuration.fix_counts();
        new_configuration.mutate_increment(0.1);
        new_configuration.mutate_targets(0.2);
        new_configuration.fix_targets();
        new_configuration.mutate_insert_pos(0.2);
    }
    return new_configuration;
}

void GAConfiguration::mutate_increment(float probability){
    std::transform(m_moves.begin(), m_moves.end(), m_moves.begin(),[this,probability](MoveTuple& mp) {
        return roll(probability) ? MoveTuple{mp.m_count,mp.m_value, !mp.m_increment} : mp;
    });
}

void GAConfiguration::mutate_insert_pos(float probability) noexcept{
    if (roll(probability))
        m_insert_pos = m_insert_dist(m_rng);
}

void GAConfiguration::mutate_counts(float probability){
    std::transform(m_moves.begin(), m_moves.end(), m_moves.begin(),[this,probability](MoveTuple& mp) {
        return roll(probability) ? MoveTuple{(uint8_t)m_count_dist(m_rng),mp.m_value,mp.m_increment} : mp;
    });
}

void GAConfiguration::read_from_file(const std::string& filename){
    std::ifstream ifs(filename,std::ios::binary);
    if (!ifs)
        throw PluginError("Can't open GA configuration file: " + filename);
    uint32_t in_line_size;
    ifs.read((char*)&in_line_size,sizeof(in_line_size));
    if (in_line_size != m_line_size)
        throw PluginError("Invalid GA configuration line length. Config = "s + std::to_string(in_line_size) + ", Cache = " + std::to_string(m_line_size));
    ifs.read((char*)&m_insert_pos,sizeof(m_insert_pos));
    for(uint32_t i = 0; i < m_line_size/4; i++)
        ifs.read((char*)&m_moves[i],sizeof(m_moves[0]));
    if (!ifs)
        throw PluginError("Invalid GA configuration file: " + filename);
}

void GAConfiguration::write_to_file(const std::string& filename) const {
    std::ofstream ofs(filename,std::ios::binary);
    if (!ofs)
        throw PluginError("Can't open GA configuration savefile: " + filename);
    ofs.write((char*)&m_line_size,sizeof(m_line_size));
    ofs.write((char*)&m_insert_pos,sizeof(m_insert_pos));
    for(uint32_t i = 0; i < m_line_size/4; i++)
        ofs.write((char*)&  m_moves[i],sizeof(m_moves[0]));
    if (!ofs)
        throw PluginError("Can't save to GA configuration file: " + filename);
}

void GAConfiguration::mutate_targets(float probability){
    uint32_t max = 0;
    std::transform(m_moves.begin(), m_moves.end(), m_moves.begin(),[this,probability,&max](MoveTuple& mp) {
        std::uniform_int_distribution<std::mt19937::result_type> dist(0,max);
        if (roll(probability))
            mp.m_value = dist(m_rng);
        //if (mp.m_count >= max)
        //    mp.m_value = max;
        max += mp.m_count;
        return mp;
    });
}

void GAConfiguration::fix_counts() noexcept{
    while (uint8_t diff = std::accumulate(m_moves.begin(), m_moves.end(), 0,[](uint8_t sum, const MoveTuple& mp) { return sum + mp.m_count; }) -  m_line_size)
        m_moves[m_pair_dist(m_rng)].m_count += diff > 0 ? 1 : -1;
}
void GAConfiguration::fix_targets() noexcept{
    for( uint32_t i = 1; i < m_moves.size(); i++)
        if (m_moves[i-1].m_value + m_moves[i-1].m_count * m_moves[i-1].m_increment > m_moves[i].m_value) {
            if (m_moves[i - 1].m_increment)
                m_moves[i].m_value = m_moves[i - 1].m_value + m_moves[i - 1].m_count * m_moves[i - 1].m_increment;
            else
                std::swap(m_moves[i].m_value, m_moves[i - 1].m_value);
        }
}

bool GAConfiguration::roll(double probability){
    if (probability > 1 || probability < 0)
        throw PluginError("Probability is not inside [0;1]");

    return m_probability_dist(m_rng)/1000.0 <= probability;
}

bool GAConfiguration::operator==(const GAConfiguration& o) const noexcept{
    return m_moves == o.m_moves && m_insert_pos == o.m_insert_pos;
}
bool GAConfiguration::operator!=(const GAConfiguration& o) const noexcept{
    return !(*this == o);
}
std::pair<uint32_t,std::vector<uint32_t>> GAConfiguration::unpack() const noexcept{
    std::vector<uint32_t> res;
    for(const auto& mt : m_moves)
        for(uint32_t i = 0; i < mt.m_count; i++)
            res.push_back(mt.m_value + ( mt.m_increment ? i : 0 ) );
    return {m_insert_pos,res};
}
} // namespace ipxp