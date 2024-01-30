//
// Created by zaida on 27.01.2024.
//

#include "movetuple.hpp"

namespace ipxp {
std::istream& operator>>(std::istream& is, MoveTuple& mp) {
    return is >> mp.m_count >> mp.m_value >> mp.m_increment;
}
std::ostream& operator<<(std::ostream& os, const MoveTuple& mp) {
    return os << mp.m_count << mp.m_value << mp.m_increment;
}
bool MoveTuple::operator==(const MoveTuple& o) const noexcept{
    return m_count == o.m_count && m_value == o.m_value && m_increment == o.m_increment;
}

} // namespace ipxp