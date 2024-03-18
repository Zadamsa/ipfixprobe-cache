//
// Created by zaida on 18.03.2024.
//

#include "qtable.hpp"

namespace ipxp {
QTable::QTable(const std::string& filename, uint32_t line_size):
    m_fs(filename, std::ios::in | std::ios::out | std::ios::binary),
    m_bucket_count(line_size/4),
    m_actions_count(m_bucket_count * 4 + 5 * 2)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
        create_empty_table();


}

void QTable::create_empty_table(){

}
} // namespace ipxp