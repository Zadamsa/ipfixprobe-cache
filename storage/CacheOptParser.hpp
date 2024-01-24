//
// Created by zaida on 24.01.2024.
//

#ifndef IPFIXPROBE_CACHE_CACHEOPTPARSER_H
#define IPFIXPROBE_CACHE_CACHEOPTPARSER_H

namespace ipxp {
class CacheOptParser : public OptionsParser {
public:
    uint32_t m_cache_size;
    uint32_t m_line_size;
    uint32_t m_active = 300;
    uint32_t m_inactive = 30;
    bool m_split_biflow;

    CacheOptParser();
};
};
#endif // IPFIXPROBE_CACHE_CACHEOPTPARSER_H
