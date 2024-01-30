//
// Created by zaida on 26.01.2024.
//

#include <ipfixprobe/utils.hpp>
#include "gaelaborationcacheoptparser.hpp"

namespace ipxp {
GAElaborationCacheOptParser::GAElaborationCacheOptParser(const char* name, const char* description): GACacheOptParser(name,description){
    register_options();
}
GAElaborationCacheOptParser::GAElaborationCacheOptParser(): GACacheOptParser("gaelabcache","Pseudo storage plugin, evaluates one genetic algorithm generation"){
    register_options();
}
void GAElaborationCacheOptParser::register_options(){
    register_option("oc", "outputconfig", "PATH", "Path to the result genetic configuration file", [this](const char *arg){m_outfilename = arg; return true;}, OptionFlags::RequiredArgument);
    register_option("g", "gensize", "SIZE", "Number of simultaneously evaluated samples", [this](const char *arg){try {
                m_generation_size = str2num<decltype(m_generation_size)>(arg);
            } catch(std::invalid_argument &e) {
                return false;
            }
            return true;
        }, OptionFlags::RequiredArgument);
}
} // namespace ipxp