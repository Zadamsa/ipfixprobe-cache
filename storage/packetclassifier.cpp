#include "packetclassifier.hpp"
namespace ipxp {
//Prijima atributy paketu. Navratova hodnota je vzdalenost dalsiho paketu v tom samem flow
PacketDistance PacketClassifier::classifyInstance(uint8_t protocol,uint8_t tcp_flags, uint16_t tcp_window, uint16_t total_length) {
    if (tcp_flags <= 16)
        if (total_length <= 35844)
            if (tcp_flags <= 4)
                if (protocol <= 6)
                    return  DISTANCE_NEVER ;
                else
                    if (protocol <= 17)
                    if (total_length <= 15361)
                        if (total_length <= 7429)
                            if (total_length <= 2561)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 11265)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 12293)
                            return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        if (total_length <= 15365)
                        return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 26884)
                        if (total_length <= 19456)
                            if (total_length <= 18180)
                                if (total_length <= 17669)
                                    if (total_length <= 16133)
                                        return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 17157)
                                        if (total_length <= 16389)
                                            return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 18432)
                                return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 19202)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 23553)
                            if (total_length <= 22532)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 23808)
                            return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        if (total_length <= 26885)
                        return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_NEVER ;
            else
                if (total_length <= 2565)
                return  DISTANCE_SHORT ;
            else
                if (tcp_window <= 2051)
                if (tcp_window <= 512)
                    if (total_length <= 10241)
                        if (tcp_window <= 486)
                            return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_SHORT ;
            else
                return  DISTANCE_SHORT ;
        else
            if (tcp_flags <= 0)
            if (total_length <= 60419)
                if (total_length <= 56325)
                    if (total_length <= 55299)
                        if (total_length <= 40961)
                            if (total_length <= 38405)
                                if (total_length <= 36867)
                                    return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 39684)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 51205)
                            if (total_length <= 41216)
                                return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 43011)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 52227)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 52228)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 54019)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 55300)
                        return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 56324)
                        return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    if (total_length <= 58625)
                    return  DISTANCE_MEDIUM ;
                else
                    if (total_length <= 58628)
                    return  DISTANCE_SHORT ;
                else
                    if (total_length <= 59648)
                    return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (total_length <= 60420)
                return  DISTANCE_SHORT ;
            else
                if (total_length <= 63491)
                return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_SHORT ;
        else
            return  DISTANCE_SHORT ;
    else
        if (tcp_flags <= 20)
        if (tcp_window <= 24083)
            return  DISTANCE_NEVER ;
        else
            return  DISTANCE_MEDIUM ;
    else
        if (total_length <= 30723)
        if (total_length <= 17920)
            if (tcp_window <= 500)
                return  DISTANCE_MEDIUM ;
            else
                if (tcp_window <= 2550)
                if (tcp_window <= 2399)
                    if (tcp_window <= 925)
                        return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                return  DISTANCE_MEDIUM ;
        else
            if (tcp_window <= 176)
            return  DISTANCE_MEDIUM ;
        else
            if (total_length <= 20997)
            if (tcp_window <= 508)
                return  DISTANCE_SHORT ;
            else
                return  DISTANCE_MEDIUM ;
        else
            return  DISTANCE_MEDIUM ;
    else
        if (total_length <= 30725)
        return  DISTANCE_MEDIUM ;
    else
        if (tcp_window <= 500)
        if (tcp_window <= 119)
            return  DISTANCE_SHORT ;
        else
            return  DISTANCE_MEDIUM ;
    else
        if (tcp_window <= 1021)
        if (tcp_window <= 1020)
            if (tcp_window <= 516)
                if (total_length <= 35845)
                    return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_MEDIUM ;
        else
            return  DISTANCE_SHORT ;
    else
        if (tcp_window <= 3483)
        return  DISTANCE_MEDIUM ;
    else
        if (total_length <= 56323)
        if (tcp_window <= 16382)
            return  DISTANCE_MEDIUM ;
        else
            return  DISTANCE_SHORT ;
    else
        return  DISTANCE_SHORT ;
}
} // namespace ipxp