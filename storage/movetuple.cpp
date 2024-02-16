#include "movetuple.hpp"

namespace ipxp {
std::istream& operator>>(std::istream& is, MoveTuple& mp) {
    return is >> mp.m_count >> mp.m_target >> mp.m_increment;
}
std::ostream& operator<<(std::ostream& os, const MoveTuple& mp) {
    return os << mp.m_count << mp.m_target << mp.m_increment;
}
bool MoveTuple::operator==(const MoveTuple& o) const noexcept{
    return m_count == o.m_count && m_target == o.m_target && m_increment == o.m_increment;
}

} // namespace ipxp