//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_CACHESTATISTICS_HPP
#define IPFIXPROBE_CACHE_CACHESTATISTICS_HPP

namespace ipxp {

struct CacheStatistics {
    uint32_t m_empty = 0;
    uint32_t m_not_empty = 0;
    uint32_t m_hits = 0;
    uint32_t m_expired = 0;
    uint32_t m_flushed = 0;
    uint32_t m_lookups = 0;
    uint32_t m_lookups2 = 0;
    CacheStatistics operator-(const CacheStatistics& o) const noexcept{
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
};
friend std::ostream& operator<<(std::ostream& os, const CacheStatistics& statistics) noexcept;
} // namespace ipxp

#endif // IPFIXPROBE_CACHE_CACHESTATISTICS_HPP
