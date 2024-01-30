//
// Created by zaida on 27.01.2024.
//

#ifndef IPFIXPROBE_CACHE_MOVETUPLE_HPP
#define IPFIXPROBE_CACHE_MOVETUPLE_HPP

#include <cstdint>
#include <istream>
#include <ostream>

namespace ipxp {

struct MoveTuple{
    uint8_t m_count;
    uint8_t m_value;
    bool m_increment = false;
    MoveTuple& operator=(const MoveTuple& o) noexcept = default;
    bool operator==(const MoveTuple& o) const noexcept;
    friend std::istream& operator>>(std::istream& is, MoveTuple& mp);
    friend std::ostream& operator<<(std::ostream& os, const MoveTuple& mp);
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_MOVETUPLE_HPP
