//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_CACHESTATISTICS_HPP
#define IPFIXPROBE_CACHE_CACHESTATISTICS_HPP

#include <cstdint>
#include <ostream>
namespace ipxp {
struct CacheStatistics {
    uint32_t m_empty = 0;
    uint32_t m_not_empty = 0;
    uint32_t m_hits = 0;
    uint32_t m_expired = 0;
    uint32_t m_flushed = 0;
    uint32_t m_lookups = 0;
    uint32_t m_lookups2 = 0;
    uint32_t m_put_time = 0;
    CacheStatistics operator-(const CacheStatistics& o) const noexcept;
    friend std::ostream& operator<<(std::ostream& os, const CacheStatistics& statistics) noexcept;
};
} // namespace ipxp

#endif // IPFIXPROBE_CACHE_CACHESTATISTICS_HPP
