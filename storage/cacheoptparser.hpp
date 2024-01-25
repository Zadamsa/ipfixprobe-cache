//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_CACHEOPTPARSER_H
#define IPFIXPROBE_CACHE_CACHEOPTPARSER_H
#include <cstdint>
#include <ipfixprobe/options.hpp>

#ifdef IPXP_FLOW_CACHE_SIZE
static const uint32_t DEFAULT_FLOW_CACHE_SIZE = IPXP_FLOW_CACHE_SIZE;
#else
static const uint32_t DEFAULT_FLOW_CACHE_SIZE = 17; // 131072 records total
#endif /* IPXP_FLOW_CACHE_SIZE */

#ifdef IPXP_FLOW_LINE_SIZE
static const uint32_t DEFAULT_FLOW_LINE_SIZE = IPXP_FLOW_LINE_SIZE;
#else
static const uint32_t DEFAULT_FLOW_LINE_SIZE = 4; // 16 records per line
#endif /* IPXP_FLOW_LINE_SIZE */

namespace ipxp {
class CacheOptParser : public OptionsParser {
public:
    CacheOptParser();

    uint32_t m_cache_size = 1 << DEFAULT_FLOW_CACHE_SIZE;
    uint32_t m_line_size = 1 << DEFAULT_FLOW_LINE_SIZE;
    uint32_t m_active = 300;
    uint32_t m_inactive = 30;
    bool m_split_biflow = false;
};
};
#endif // IPFIXPROBE_CACHE_CACHEOPTPARSER_H
