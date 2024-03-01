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

//Vytvori nahodnou(ale validni) konfiguraci
GAConfiguration::GAConfiguration(uint32_t line_size): m_line_size(line_size){
    m_rng = std::mt19937(std::random_device()());
    m_probability_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,1000);
    m_pair_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size/4 - 1);
    m_count_dist = std::uniform_int_distribution<std::mt19937::result_type>(1,m_line_size/4 );
    m_insert_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size - 1);

    uint32_t prev = 0;
    uint32_t generated_nodes_count = 0;
    for( uint32_t i = 0; i < line_size/4; i++){
        std::uniform_int_distribution<std::mt19937::result_type> moves_dist(prev,generated_nodes_count);
        uint8_t count = m_count_dist(m_rng);
        uint8_t target = moves_dist(m_rng);
        generated_nodes_count += count;
        prev = target;
        m_moves.push_back(MoveTuple{count,target, roll(0.1)});
    }
    mutate_insert_pos(1);
    fix_counts();
    fix_targets();

}

GAConfiguration::GAConfiguration(const GAConfiguration& o){
    m_moves = o.m_moves;
    m_short_pos = o.m_short_pos;
    m_medium_pos = o.m_medium_pos;
    m_long_pos = o.m_long_pos;
    m_never_pos = o.m_never_pos;
    m_line_size = o.m_line_size;
    m_rng = std::mt19937(std::random_device()());
    m_probability_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,1000);
    m_pair_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size/4 - 1);
    m_count_dist = std::uniform_int_distribution<std::mt19937::result_type>(1,m_line_size/4 );
    m_insert_dist = std::uniform_int_distribution<std::mt19937::result_type>(0,m_line_size - 1);
}

GAConfiguration& GAConfiguration::operator=(const GAConfiguration& o) noexcept{
    m_moves = o.m_moves;
    m_short_pos = o.m_short_pos;
    m_medium_pos = o.m_medium_pos;
    m_long_pos = o.m_long_pos;
    m_never_pos = o.m_never_pos;
    m_line_size = o.m_line_size;
    return *this;
}

GAConfiguration::GAConfiguration(){};

//Vytvori mutace konfiguraci, puvodni konfigurace zustane bez zmen
GAConfiguration GAConfiguration::mutate() const{
    GAConfiguration new_configuration = *this;
    while(new_configuration == *this) {
        new_configuration.mutate_counts(0.2);
        new_configuration.fix_counts();
        new_configuration.mutate_counts_by_one(0.4);
        new_configuration.fix_counts();
        new_configuration.mutate_increment(0.3);
        new_configuration.fix_targets();
        new_configuration.mutate_targets(0.2);
        new_configuration.fix_targets();
        new_configuration.mutate_targets_by_one(0.4);
        new_configuration.fix_targets();
        new_configuration.mutate_insert_pos(0.2);
    }
    /*if (new_configuration.is_not_valid()){
        new_configuration.write_to_file("invalid_gaconfig.bin");
        std::cout << "Invalid configuration generated" << std::endl;
        *(uint32_t*)0=6666;
    }*/
    return new_configuration;
}

//Debugovaci funkce, vrati true, pokud konfigurace je navalidni
bool GAConfiguration::is_not_valid() const noexcept{
    if (std::accumulate(m_moves.begin(), m_moves.end(), 0u,[](uint32_t sum, const MoveTuple& mp) { return sum + mp.m_count; }) > m_line_size)
        return true;
    for(const auto& mt : m_moves)
        if (mt.m_target >= m_line_size)
            return true;
    return false;
}

//Mutace poctu elemntu v MoveTuple o jedna
void GAConfiguration::mutate_counts_by_one(float probability) {
    std::transform(m_moves.begin(), m_moves.end(), m_moves.begin(),[this,probability](MoveTuple& mp) {
        return roll(probability) ? MoveTuple{std::max(mp.m_count + (roll(0.5) ? 1 : -1),1u),mp.m_target,mp.m_increment} : mp;
    });
}

//Mutace cilu posuvu o jedna
void GAConfiguration::mutate_targets_by_one(float probability) {
    if (m_moves.size() <= 1)
        return ;
    // Prvni MoveTuple vzdy musi mit za cil posuvu 0, jinak by konfigurace nevyuzivala celou delku radku
    auto second = ++m_moves.begin();
    auto last = std::prev(m_moves.end());
    std::transform(second, last, second,[this,probability](MoveTuple& mp) {
        if (mp.m_target == 0)
            return roll(probability) ? MoveTuple{mp.m_count, 1,mp.m_increment} : mp;
        if (mp.m_target == m_line_size - 1 && roll(probability))
            return MoveTuple{mp.m_count, m_line_size - 2,mp.m_increment};
        return roll(probability) ? MoveTuple{mp.m_count, mp.m_target + (roll(0.5) ? 1 : -1),mp.m_increment} : mp;
    });
    if (roll(probability))
        last->m_target--;
}

void GAConfiguration::mutate_increment(float probability){
    auto second = ++m_moves.begin();
    std::transform(second, m_moves.end(), second,[this,probability](MoveTuple& mp) {
        return roll(probability) ? MoveTuple{mp.m_count,mp.m_target, !mp.m_increment} : mp;
    });
}

// Mutace pozice vkladani novych flow
void GAConfiguration::mutate_insert_pos(float probability){
    if (roll(probability))
        m_insert_pos = m_insert_dist(m_rng);

    if (roll(probability))
        m_short_pos = m_offset_dist(m_rng) - m_line_size/2;

    std::uniform_int_distribution<std::mt19937::result_type> medium_dist(m_short_pos,m_line_size/2);
    if (roll(probability))
        m_medium_pos = medium_dist(m_rng);
    else
        m_medium_pos = std::max(m_medium_pos,m_short_pos);

    std::uniform_int_distribution<std::mt19937::result_type> long_dist(m_medium_pos,m_line_size/2);
    if (roll(probability))
        m_long_pos = long_dist(m_rng);
    else
        m_long_pos = std::max(m_long_pos,m_medium_pos);

    std::uniform_int_distribution<std::mt19937::result_type> never_dist(m_long_pos,m_line_size/2);
    if (roll(probability))
        m_never_pos = never_dist(m_rng);
    else
        m_never_pos = std::max(m_never_pos,m_long_pos);
}

// Nahodna mutace poctu flow v MoveTuple
void GAConfiguration::mutate_counts(float probability){
    std::transform(m_moves.begin(), m_moves.end(), m_moves.begin(),[this,probability](MoveTuple& mp) {
        return roll(probability) ? MoveTuple{(uint32_t)m_count_dist(m_rng),mp.m_target,mp.m_increment} : mp;
    });
}

// Nacist konfigurace ze souboru
void GAConfiguration::read_from_file(const std::string& filename){
    std::ifstream ifs(filename,std::ios::binary);
    if (!ifs)
        throw PluginError("Can't open GA configuration file: " + filename);
    uint32_t in_line_size;
    ifs.read((char*)&in_line_size,sizeof(in_line_size));
    if (in_line_size != m_line_size)
        throw PluginError("Invalid GA configuration line length. Config = "s + std::to_string(in_line_size) + ", Cache = " + std::to_string(m_line_size));
    ifs.read((char*)&m_insert_pos,sizeof(m_insert_pos));
    ifs.read((char*)&m_short_pos,sizeof(m_short_pos));
    ifs.read((char*)&m_medium_pos,sizeof(m_medium_pos));
    ifs.read((char*)&m_long_pos,sizeof(m_long_pos));
    ifs.read((char*)&m_never_pos,sizeof(m_never_pos));
    for(uint32_t i = 0; i < m_line_size/4; i++)
        ifs.read((char*)&m_moves[i],sizeof(m_moves[0]));
    if (!ifs)
        throw PluginError("Invalid GA configuration file: " + filename);
}

// Ulozit konfigurace do souboru
void GAConfiguration::write_to_file(const std::string& filename) const {
    std::ofstream ofs(filename,std::ios::binary);
    if (!ofs)
        throw PluginError("Can't open GA configuration savefile: " + filename);
    ofs.write((char*)&m_line_size,sizeof(m_line_size));
    ofs.write((char*)&m_insert_pos,sizeof(m_insert_pos));
    ofs.write((char*)&m_short_pos,sizeof(m_short_pos));
    ofs.write((char*)&m_medium_pos,sizeof(m_medium_pos));
    ofs.write((char*)&m_long_pos,sizeof(m_long_pos));
    ofs.write((char*)&m_never_pos,sizeof(m_never_pos));
    for(uint32_t i = 0; i < m_line_size/4; i++)
        ofs.write((char*)&  m_moves[i],sizeof(m_moves[0]));
    if (!ofs)
        throw PluginError("Can't save to GA configuration file: " + filename);
}

// Nahodna mutace cilu posuvu
void GAConfiguration::mutate_targets(float probability){
    uint32_t max = m_moves.begin()->m_count;
    // Prvni MoveTuple vzdy musi mit za cil posuvu 0, jinak by konfigurace nevyuzivala celou delku radku
    auto second = ++m_moves.begin();
    std::transform(second, m_moves.end(), second,[this,probability,&max](MoveTuple& mp) {
        std::uniform_int_distribution<std::mt19937::result_type> dist(0,max - 1);
        if (roll(probability)) {
            mp.m_target = dist(m_rng);
        }
        max += mp.m_count;
        return mp;
    });
}

// Po mutaci konfigurace bude nejspis v nevalidnim stavu, funkce opravi pocty prvku
void GAConfiguration::fix_counts() noexcept{
    // Oprava bude udelana tak, ze se budou nahodne volit MoveTuple a m_count bude zvysen nebo zmensen o jedna
    while (int32_t diff = std::accumulate(m_moves.begin(), m_moves.end(), 0,[](int32_t sum, const MoveTuple& mp) { return sum + (int32_t)mp.m_count; }) -  (int32_t)m_line_size)
        if (auto pos = m_pair_dist(m_rng); m_moves[pos].m_count != 1 || diff < 0)
            m_moves[pos].m_count += diff > 0 ? -1 : 1;
}

// Oprava cilu posuvu
void GAConfiguration::fix_targets() noexcept{
    for( uint32_t i = 1; i < m_moves.size(); i++)
        // Konfigurace nebude optimalni pokud i<j && conf[i] > conf[j], kde conf[i] je pozice, kam se ma posunout i-y flow
        if (m_moves[i-1].m_target + m_moves[i-1].m_count * m_moves[i-1].m_increment > m_moves[i].m_target) {
            if (m_moves[i - 1].m_increment)
                // Pokud incrementalni, nastav conf[j] na minimalni moznou hodnotu
                m_moves[i].m_target = m_moves[i - 1].m_target + m_moves[i - 1].m_count * m_moves[i - 1].m_increment;
            else
                // Pokud neni incrementalni, vymeni conf[i] a conf[j]
                std::swap(m_moves[i].m_target, m_moves[i - 1].m_target);
        }
    uint32_t max = m_moves.begin()->m_count;
    auto second = ++m_moves.begin();
    std::transform(second, m_moves.end(), second,[this,&max](MoveTuple& mp) {
        if (mp.m_target > max - 1)
            mp.m_target = max - 1;
        max += mp.m_count;
        return mp;
    });
}

//Vrati true s pravdepodobnosti probability
bool GAConfiguration::roll(double probability){
    if (probability > 1 || probability < 0)
        throw PluginError("Probability is not inside [0;1]");

    return m_probability_dist(m_rng)/1000.0 <= probability;
}

bool GAConfiguration::operator==(const GAConfiguration& o) const noexcept{
    return m_moves == o.m_moves && std::tie(m_short_pos,m_medium_pos,m_long_pos,m_never_pos) == std::tie(o.m_short_pos,o.m_medium_pos,o.m_long_pos,o.m_never_pos);
}
bool GAConfiguration::operator!=(const GAConfiguration& o) const noexcept{
    return !(*this == o);
}

//Prevede configurace do formy, kde unpacked_configuration[i] = na jakou pozice se ma posunout i-y flow
std::tuple<uint32_t,int32_t,int32_t,int32_t,int32_t,std::vector<uint32_t>> GAConfiguration::unpack() const noexcept{
    std::vector<uint32_t> res;
    for(const auto& mt : m_moves)
        for(uint32_t i = 0; i < mt.m_count; i++)
            res.push_back(mt.m_target + ( mt.m_increment ? i : 0 ) );
    return {m_insert_pos,m_short_pos,m_medium_pos,m_long_pos,m_never_pos,res};
}

std::string GAConfiguration::to_string() const noexcept{
    std::string res = std::to_string(m_insert_pos) +":"+std::to_string(m_short_pos) +"^"+std::to_string(m_medium_pos)+"^"+std::to_string(m_long_pos)+"^"+std::to_string(m_never_pos)+"{";
    for(const auto& mp: m_moves){
        if (&mp != &m_moves.front())
            res += ',';
        res += "<c:"s + std::to_string(mp.m_count) + ", ";
        res += "v:"s + std::to_string(mp.m_target) + ", ";
        res += "i:"s + std::to_string(mp.m_increment) + " >";
    }
    res += "}";
    return res;
}
} // namespace ipxp