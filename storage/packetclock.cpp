
#include "packetclock.hpp"

namespace ipxp {
    PacketClock::time_point PacketClock::now() noexcept{
        while(!PacketClock::m_started);
        return  PacketClock::m_current_time;
    }
    timeval  PacketClock::now_as_timeval() noexcept{
        while(!PacketClock::m_started);
        return  PacketClock::m_current_time_tv;
    }
    void  PacketClock::set_time(timeval tv) noexcept{
	auto new_val = time_point(std::chrono::duration_cast<duration>(std::chrono::seconds(tv.tv_sec) + std::chrono::microseconds(tv.tv_usec)));
        if (new_val <= PacketClock::m_current_time)
		return;
	PacketClock::m_current_time_tv = tv;
        PacketClock::m_current_time = new_val;
	PacketClock::m_started = true;
    }
    void  PacketClock::stop() noexcept{
        PacketClock::m_stopped = true;
    	PacketClock::m_current_time += std::chrono::seconds(2);
    }
    bool  PacketClock::has_stopped() noexcept{
        return PacketClock::m_stopped;
    }


} // namespace ipxp
