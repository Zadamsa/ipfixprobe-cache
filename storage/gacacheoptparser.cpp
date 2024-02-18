//
// Created by zaida on 26.01.2024.
//

#include "gacacheoptparser.hpp"

namespace ipxp {

GACacheOptParser::GACacheOptParser(): CacheOptParser("gacache","Storage plugin implemented as a hash table, every row is managed by genetic algorithm"){
    register_options();
}

GACacheOptParser::GACacheOptParser(const char* name, const char* description ): CacheOptParser(name,description){
    register_options();
}

void GACacheOptParser::register_options(){
    register_option("ic", "inputconfig", "PATH", "Path to the genetic configuration file", [this](const char *arg){m_infilename = arg; return true;}, OptionFlags::RequiredArgument);
}

} // namespace ipxp