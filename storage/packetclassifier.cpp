#include "packetclassifier.hpp"
namespace ipxp {
//Prijima atributy paketu. Navratova hodnota je vzdalenost dalsiho paketu v tom samem flow
PacketDistance PacketClassifier::classifyInstance(uint8_t protocol,uint8_t tcp_flags, uint16_t tcp_window, uint16_t total_length) {
    if (tcp_flags <= 4)
        if (tcp_window <= 0)
            if (protocol <= 6)
                if (total_length <= 12524)
                    if (total_length <= 9708)
                        return  PacketDistance::DISTANCE_LONG ;
                    else
                        return  PacketDistance::DISTANCE_SHORT ;
                else
                    return  PacketDistance::DISTANCE_SHORT ;
            else
                if (total_length <= 38639)
                if (total_length <= 8940)
                    if (total_length <= 7153)
                        if (total_length <= 7151)
                            return  PacketDistance::DISTANCE_MEDIUM ;
                        else
                            return  PacketDistance::DISTANCE_SHORT ;
                    else
                        return  PacketDistance::DISTANCE_MEDIUM ;
                else
                    if (total_length <= 26609)
                    if (total_length <= 22769)
                        if (total_length <= 21999)
                            return  PacketDistance::DISTANCE_MEDIUM ;
                        else
                            return  PacketDistance::DISTANCE_SHORT ;
                    else
                        return  PacketDistance::DISTANCE_MEDIUM ;
                else
                    if (total_length <= 26865)
                    return  PacketDistance::DISTANCE_SHORT ;
                else
                    return  PacketDistance::DISTANCE_MEDIUM ;
            else
                if (total_length <= 60399)
                if (total_length <= 49391)
                    return  PacketDistance::DISTANCE_MEDIUM ;
                else
                    if (total_length <= 52209)
                    if (total_length <= 51184)
                        return  PacketDistance::DISTANCE_MEDIUM ;
                    else
                        if (total_length <= 51185)
                        return  PacketDistance::DISTANCE_SHORT ;
                    else
                        if (total_length <= 52207)
                        return  PacketDistance::DISTANCE_MEDIUM ;
                    else
                        return  PacketDistance::DISTANCE_SHORT ;
                else
                    return  PacketDistance::DISTANCE_MEDIUM ;
            else
                if (total_length <= 60400)
                return  PacketDistance::DISTANCE_SHORT ;
            else
                return  PacketDistance::DISTANCE_MEDIUM ;
        else
            return  PacketDistance::DISTANCE_NEVER ;
    else
        if (tcp_flags <= 16)
        if (tcp_window <= 1026)
            if (total_length <= 18924)
                if (tcp_window <= 1017)
                    return  PacketDistance::DISTANCE_SHORT ;
                else
                    if (tcp_window <= 1025)
                    if (tcp_window <= 1023)
                        return  PacketDistance::DISTANCE_MEDIUM ;
                    else
                        return  PacketDistance::DISTANCE_SHORT ;
                else
                    return  PacketDistance::DISTANCE_SHORT ;
            else
                return  PacketDistance::DISTANCE_SHORT ;
        else
            if (tcp_window <= 4152)
            if (tcp_window <= 4140)
                return  PacketDistance::DISTANCE_SHORT ;
            else
                return  PacketDistance::DISTANCE_MEDIUM ;
        else
            return  PacketDistance::DISTANCE_SHORT ;
    else
        if (tcp_flags <= 20)
        return  PacketDistance::DISTANCE_MEDIUM ;
    else
        if (total_length <= 51183)
        if (tcp_window <= 1026)
            if (tcp_window <= 869)
                if (tcp_window <= 154)
                    if (tcp_window <= 8)
                        return  PacketDistance::DISTANCE_MEDIUM ;
                    else
                        return  PacketDistance::DISTANCE_SHORT ;
                else
                    if (tcp_window <= 298)
                    return  PacketDistance::DISTANCE_MEDIUM ;
                else
                    if (total_length <= 23025)
                    return  PacketDistance::DISTANCE_SHORT ;
                else
                    return  PacketDistance::DISTANCE_MEDIUM ;
            else
                return  PacketDistance::DISTANCE_MEDIUM ;
        else
            if (tcp_window <= 1044)
            return  PacketDistance::DISTANCE_SHORT ;
        else
            if (total_length <= 20721)
            return  PacketDistance::DISTANCE_MEDIUM ;
        else
            if (total_length <= 26860)
            return  PacketDistance::DISTANCE_SHORT ;
        else
            if (tcp_window <= 16117)
            return  PacketDistance::DISTANCE_MEDIUM ;
        else
            return  PacketDistance::DISTANCE_SHORT ;
    else
        return  PacketDistance::DISTANCE_SHORT ;
}
} // namespace ipxp