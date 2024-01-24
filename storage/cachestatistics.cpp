//
// Created by zaida on 24.01.2024.
//

#include "cachestatistics.hpp"

namespace ipxp {

friend std::ostream& operator<<(std::ostream& os, const CacheStatistics& statistics) noexcept{
    float tmp = float(statistics.m_lookups) / statistics.m_hits;
    os << "Hits: " << statistics.m_hits << std::endl;
    os << "Empty: " << statistics.m_empty << std::endl;
    os << "Not empty: " << statistics.m_not_empty << std::endl;
    os << "Expired: " << statistics.m_expired << std::endl;
    os << "Flushed: " << statistics.m_flushed << std::endl;
    os << "Average Lookup:  " << tmp << std::endl;
    os << "Variance Lookup: " << float(statistics.m_lookups2) / statistics.m_hits - tmp * tmp << std::endl;
    os << "Spent in put_pkt: " << statistics.m_put_time << " us" << std::endl;
}

} // namespace ipxp