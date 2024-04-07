#ifndef CACHE_CPP_PACKETCLOCK_HPP
#define CACHE_CPP_PACKETCLOCK_HPP

#include "sys/time.h"
#include "chrono"

namespace ipxp {

class PacketClock {

public:
    using timer_type = std::chrono::steady_clock;
    using time_point = std::chrono::time_point<PacketClock, typename std::chrono::steady_clock::time_point::duration >;
    using duration   = typename std::chrono::steady_clock::duration;
    using rep    = typename duration::rep;
    using period     = typename duration::period;
    static const bool is_steady = false;

    static time_point now() noexcept{
        while(m_current_time == time_point());
        return  m_current_time;
    }
    static timeval now_as_timeval() noexcept{
        while(m_current_time == time_point());
        return  m_current_time_tv;
    }
    static void set_time(timeval tv) noexcept{
        m_current_time_tv = tv;
        m_current_time = time_point(std::chrono::duration_cast<duration>(std::chrono::seconds(tv.tv_sec) + std::chrono::microseconds(tv.tv_usec)));
    }
private:
    inline static time_point m_current_time;
    inline static timeval m_current_time_tv;
};

} // namespace ipxp

#endif // CACHE_CPP_PACKETCLOCK_HPP
