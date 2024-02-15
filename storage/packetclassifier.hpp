//
// Created by zaida on 13.02.2024.
//

#ifndef CACHE_CPP_PACKETCLASSIFIER_HPP
#define CACHE_CPP_PACKETCLASSIFIER_HPP

namespace ipxp {
enum class PacketDistance : uint8_t {
    DISTANCE_SHORT,
    DISTANCE_MEDIUM,
    DISTANCE_LONG,
    DISTANCE_NEVER
};
class PacketClassifier {
    static PacketDistance classifyInstance(uint8_t attribute0, uint16_t attribute1, uint16_t attribute2);
};

} // namespace ipxp

#endif // CACHE_CPP_PACKETCLASSIFIER_HPP
