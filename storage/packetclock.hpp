#ifndef CACHE_CPP_PACKETCLOCK_HPP
#define CACHE_CPP_PACKETCLOCK_HPP

#include "sys/time.h"
#include "chrono"
#include <algorithm>
#include <atomic>
namespace ipxp {

class PacketClock {

public:
    using timer_type = std::chrono::steady_clock;
    using time_point = std::chrono::time_point<PacketClock, typename std::chrono::steady_clock::time_point::duration >;
    using duration   = typename std::chrono::steady_clock::duration;
    using rep    = typename duration::rep;
    using period     = typename duration::period;
    static const bool is_steady = false;

    static time_point now() noexcept;
    static timeval now_as_timeval() noexcept;
    static void set_time(timeval tv) noexcept;
    static void stop() noexcept;
    static bool has_stopped() noexcept;
private:
    inline static time_point m_current_time;
    inline static timeval m_current_time_tv;
    inline static std::atomic<bool> m_stopped = false;
    inline static std::atomic<bool> m_started = false;
};

} // namespace ipxp

#endif // CACHE_CPP_PACKETCLOCK_HPP
