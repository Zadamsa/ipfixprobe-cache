#ifndef IPFIXPROBE_CACHE_MOVETUPLE_HPP
#define IPFIXPROBE_CACHE_MOVETUPLE_HPP

#include <cstdint>
#include <istream>
#include <ostream>

namespace ipxp {

// Libovolna sprava radku cache definuje jak se maji posouvat flow zaznamy jiz existujici v radku, pokud prijde
// dalsi paket a pro nejake flow nastane cache hit. Treba LRU to pravidlo posuvu ma dost jenoduche - flow s libovolne pozice
// se primisti na prvni misto a odpovidajici flow se posunou smerem ke konci radku. Pokud pro jednoduchost zvolime delku radku = 8
// muzeme zakodovat LRU ciselne jako 0,0,0,0,0,0,0,0 - cache hit na libovolne pozice vede na presun na prvni(nultou) pozice.
// Podobne muzeme zakodovat treba 0,0,1,1,2,4,5,6 - coz by znamenalo ze flow z pozice 0 a 1 se posune na nultou pozice, 2 a 3 na prvni pozice
// 4 na 2ou, 5 na 4ou,6 na 5ou pozici, a flow cislo 7 na sestou pozici. To same muzeme zapsat usporneji jako
// {0,2,false},{1,2,false},{2,1,false/true},{4,3,true}
// Structura definuje prave to co je v zavorkach {m_target,m_count,m_increment}
struct MoveTuple{
    uint32_t m_count;
    uint32_t m_target;
    bool m_increment = false;
    MoveTuple& operator=(const MoveTuple& o) noexcept = default;
    bool operator==(const MoveTuple& o) const noexcept;
    friend std::istream& operator>>(std::istream& is, MoveTuple& mp);
    friend std::ostream& operator<<(std::ostream& os, const MoveTuple& mp);
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_MOVETUPLE_HPP
