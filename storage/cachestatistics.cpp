//
// Created by zaida on 24.01.2024.
//

#include "cachestatistics.hpp"

namespace ipxp {

CacheStatistics CacheStatistics::operator-(const CacheStatistics& o) const noexcept{
    CacheStatistics res;
    res.m_empty = m_empty - o.m_empty;
    res.m_not_empty = m_not_empty - o.m_not_empty;
    res.m_hits = m_hits - o.m_hits;
    res.m_expired = m_expired - o.m_expired;
    res.m_flushed = m_flushed - o.m_flushed;
    res.m_lookups = m_lookups - o.m_lookups;
    res.m_lookups2 = m_lookups2 - o.m_lookups2;
    return res;
}
std::ostream& operator<<(std::ostream& os, const CacheStatistics& statistics) noexcept{
    os << "==================================================================\n";
    float tmp = float(statistics.m_lookups) / statistics.m_hits;
    os << "Hits: " << statistics.m_hits << "\n";
    os << "Empty: " << statistics.m_empty << "\n";
    os << "Not empty: " << statistics.m_not_empty << "\n";
    os << "Expired: " << statistics.m_expired << "\n";
    os << "Flushed: " << statistics.m_flushed << "\n";
    os << "Average Lookup:  " << tmp << "\n";
    os << "Variance Lookup: " << float(statistics.m_lookups2) / statistics.m_hits - tmp * tmp << "\n";
    os << "Spent in put_pkt: " << statistics.m_put_time << " us" << std::endl;
    return os;
}

} // namespace ipxp