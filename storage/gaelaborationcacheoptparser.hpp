//
// Created by zaida on 26.01.2024.
//

#ifndef IPFIXPROBE_CACHE_GAELABORATIONCACHEOPTPARSER_HPP
#define IPFIXPROBE_CACHE_GAELABORATIONCACHEOPTPARSER_HPP

#include "gacacheoptparser.hpp"

namespace ipxp {

class GAElaborationCacheOptParser : public GACacheOptParser{
public:
    GAElaborationCacheOptParser();
    GAElaborationCacheOptParser(const char* name, const char* description);
    std::string m_outfilename = ""; ///< Savefile for best found configuration
    uint8_t m_generation_size = 16; ///< Count of tested caches
private:
    void register_options();
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GAELABORATIONCACHEOPTPARSER_HPP
