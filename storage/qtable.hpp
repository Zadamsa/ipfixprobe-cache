#ifndef CACHE_CPP_QTABLE_HPP
#define CACHE_CPP_QTABLE_HPP
#include "fstream"
namespace ipxp {

class QTable {
public:
    struct QTableRecord{
        uint32_t m_not_empty = 0;
        bool m_empty = true;
    };
    const uint32_t m_bucket_count = 0;
    const uint32_t m_actions_count = 0;
private:
    std::fstream m_fs;

};

} // namespace ipxp

#endif // CACHE_CPP_QTABLE_HPP
