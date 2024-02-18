//
// Created by zaida on 26.01.2024.
//

#ifndef IPFIXPROBE_CACHE_GACACHEOPTPARSER_HPP
#define IPFIXPROBE_CACHE_GACACHEOPTPARSER_HPP

#include <string>
#include "cacheoptparser.hpp"

namespace ipxp {

class GACacheOptParser : public CacheOptParser {
public:
    GACacheOptParser();
    GACacheOptParser(const char* name, const char* description);
    std::string m_infilename = ""; ///< Name of file, that contains GA configuration
private:
    void register_options();
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_GACACHEOPTPARSER_HPP
