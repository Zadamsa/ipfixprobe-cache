//
// Created by zaida on 24.01.2024.
//

#include "cacheoptparser.hpp"
#include <ipfixprobe/utils.hpp>
#include <ipfixprobe/plugin.hpp>
namespace ipxp {
CacheOptParser::CacheOptParser() : OptionsParser("cache", "Storage plugin implemented as a hash table"){
    register_options();
}
CacheOptParser::CacheOptParser(const char* name,const char* description): OptionsParser(name, description){
    register_options();
}
void CacheOptParser::register_options(){
    register_option(
        "s",
        "size",
        "EXPONENT",
        "Cache size exponent to the power of two",
        [this](const char* arg) {
            try {
                unsigned exp = str2num<decltype(exp)>(arg);
                if (exp < 4 || exp > 30) {
                    throw PluginError("Flow cache size must be between 4 and 30");
                }
                m_cache_size = static_cast<uint32_t>(1) << exp;
            } catch (std::invalid_argument& e) {
                return false;
            }
            return true;
        },
        OptionFlags::RequiredArgument);
    register_option(
        "l",
        "line",
        "EXPONENT",
        "Cache line size exponent to the power of two",
        [this](const char* arg) {
            try {
                m_line_size = static_cast<uint32_t>(1)
                    << str2num<decltype(m_line_size)>(arg);
                if (m_line_size < 1) {
                    throw PluginError("Flow cache line size must be at least 1");
                }
            } catch (std::invalid_argument& e) {
                return false;
            }
            return true;
        },
        OptionFlags::RequiredArgument);
    register_option(
        "a",
        "active",
        "TIME",
        "Active timeout in seconds",
        [this](const char* arg) {
            try {
                m_active = str2num<decltype(m_active)>(arg);
            } catch (std::invalid_argument& e) {
                return false;
            }
            return true;
        },
        OptionFlags::RequiredArgument);
    register_option(
        "i",
        "inactive",
        "TIME",
        "Inactive timeout in seconds",
        [this](const char* arg) {
            try {
                m_inactive = str2num<decltype(m_inactive)>(arg);
            } catch (std::invalid_argument& e) {
                return false;
            }
            return true;
        },
        OptionFlags::RequiredArgument);
    register_option(
        "p",
        "period",
        "TIME",
        "Print cache statistics every period of time",
        [this](const char *arg){try {
                m_periodic_statistics_sleep_time = str2num<decltype(m_periodic_statistics_sleep_time)>(arg);
            } catch(std::invalid_argument &e) {
                return false;
            }
            return true;
        },
        OptionFlags::RequiredArgument);
    register_option(
        "S",
        "split",
        "",
        "Split biflows into uniflows",
        [this](const char* arg) {
            m_split_biflow = true;
            return true;
        },
        OptionFlags::NoArgument);
}
};