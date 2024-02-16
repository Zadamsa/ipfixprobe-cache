#ifndef CACHE_CPP_PACKETCLASSIFIER_HPP
#define CACHE_CPP_PACKETCLASSIFIER_HPP

namespace ipxp {
//Enum reprezentuje vzdalenost mezi paketem a dalsim paketem, patricim stejnemu flow(v paketech)
enum class PacketDistance : uint8_t {
    DISTANCE_SHORT,//Pokud dalsi paket pro ten flow prijde pres mene nez 1000 paketu
    DISTANCE_MEDIUM,//Pokud dalsi paket pro ten flow prijde pres mene nez 100000 paketu
    DISTANCE_NEVER//Pokud paket byl posledni
};

class PacketClassifier {
    static PacketDistance classifyInstance(uint8_t tcp_flags, uint16_t tcp_window, uint16_t total_length);
};

} // namespace ipxp

#endif // CACHE_CPP_PACKETCLASSIFIER_HPP
