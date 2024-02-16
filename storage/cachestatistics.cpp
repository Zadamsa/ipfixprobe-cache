/**
 * \file cachestatistics.hpp
 * \brief Statistics about packet insertion into cache
 */
/*
 * Copyright (C) 2023 CESNET
 *
 * LICENSE TERMS
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name of the Company nor the names of its contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 */
#include <fstream>
#include <ipfixprobe/plugin.hpp>
#include "cachestatistics.hpp"

namespace ipxp {

CacheStatistics::CacheStatistics()
    : m_empty(0)
    , m_not_empty(0)
    , m_hits(0)
    , m_expired(0)
    , m_flushed(0)
    , m_lookups(0)
    , m_lookups2(0)
    , m_exported_on_cache_end(0)
    , m_exported_on_periodic_scan(0)
    , m_put_time(0)
{
}

CacheStatistics CacheStatistics::operator-(const CacheStatistics& o) const noexcept
{
    CacheStatistics res;
    res.m_empty = m_empty - o.m_empty;
    res.m_not_empty = m_not_empty - o.m_not_empty;
    res.m_hits = m_hits - o.m_hits;
    res.m_expired = m_expired - o.m_expired;
    res.m_flushed = m_flushed - o.m_flushed;
    res.m_lookups = m_lookups - o.m_lookups;
    res.m_lookups2 = m_lookups2 - o.m_lookups2;
    res.m_put_time = m_put_time - o.m_put_time;
    return res;
}
std::ostream& operator<<(std::ostream& os, const CacheStatistics& statistics) noexcept
{
    os << "==================================================================\n";
    float tmp = float(statistics.m_lookups) / statistics.m_hits;
    os << "Hits: " << statistics.m_hits << "\n";
    os << "Empty: " << statistics.m_empty << "\n";
    os << "Not empty: " << statistics.m_not_empty << "\n";
    os << "Expired: " << statistics.m_expired << "\n";
    os << "Exported on cache end: " << statistics.m_exported_on_cache_end << "\n";
    os << "Exported on periodic scan: " << statistics.m_exported_on_periodic_scan << "\n";
    os << "Flushed: " << statistics.m_flushed << "\n";
    os << "Average Lookup:  " << tmp << "\n";
    os << "Variance Lookup: " << float(statistics.m_lookups2) / statistics.m_hits - tmp * tmp << "\n";
    os << "Spent in put_pkt: " << statistics.m_put_time << " ns" << std::endl;
    return os;
}

// Pro rozhodnuti jaka sprava je lepsi porovnavaji se pocty exportovani flow z duvodu nedostatku volneho mista v radku
bool CacheStatistics::operator<(const CacheStatistics& o) const noexcept{
    return m_not_empty < o.m_not_empty;
}

void CacheStatistics::read_from_file(const std::string& filename){
    std::ifstream ifs(filename,std::ios::binary);
    if (!ifs)
        throw PluginError("Can't open GA statistics savefile: " + filename);
    ifs.read((char*)&m_not_empty,sizeof(m_not_empty));
    if (!ifs)
        throw PluginError("Invalid GA statistics file: " + filename);
}
void CacheStatistics::write_to_file(const std::string& filename) const{
    std::ofstream ofs(filename,std::ios::binary);
    if (!ofs)
        throw PluginError("Can't open GA statistics savefile: " + filename);
    ofs.write((char*)&m_not_empty,sizeof(m_not_empty));
    if (!ofs)
        throw PluginError("Can't save to GA statistics file: " + filename);
}


} // namespace ipxp