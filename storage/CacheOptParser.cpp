//
// Created by zaida on 24.01.2024.
//

#include "CacheOptParser.hpp"
namespace ipxp {
CacheOptParser::CacheOptParser()
        : OptionsParser("cache", "Storage plugin implemented as a hash table")
        , m_cache_size(1 << DEFAULT_FLOW_CACHE_SIZE)
        , m_line_size(1 << DEFAULT_FLOW_LINE_SIZE)
        , m_split_biflow(false)
        {
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