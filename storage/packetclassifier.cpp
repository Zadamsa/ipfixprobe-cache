#include "packetclassifier.hpp"
namespace ipxp {
//Prijima atributy paketu. Navratova hodnota je vzdalenost dalsiho paketu v tom samem flow
PacketDistance PacketClassifier::classifyInstance(uint8_t protocol,uint8_t tcp_flags, uint16_t tcp_window, uint16_t total_length) {
    if (tcp_flags <= 16)
        if (total_length <= 35824)
            if (tcp_flags <= 4)
                if (tcp_window <= 29)
                    if (protocol <= 6)
                        return  DISTANCE_NEVER ;
                    else
                        if (protocol <= 17)
                        if (total_length <= 15341)
                            if (total_length <= 13041)
                                if (total_length <= 12805)
                                    if (total_length <= 12013)
                                        if (total_length <= 8451)
                                            if (total_length <= 7153)
                                                if (total_length <= 7149)
                                                    if (total_length <= 3053)
                                                        if (total_length <= 2049)
                                                            return  DISTANCE_MEDIUM ;
                                                        else
                                                            if (total_length <= 2289)
                                                            return  DISTANCE_SHORT ;
                                                        else
                                                            return  DISTANCE_MEDIUM ;
                                                    else
                                                        if (total_length <= 3056)
                                                        return  DISTANCE_MEDIUM ;
                                                    else
                                                        if (total_length <= 3565)
                                                        return  DISTANCE_SHORT ;
                                                    else
                                                        return  DISTANCE_MEDIUM ;
                                                else
                                                    return  DISTANCE_SHORT ;
                                            else
                                                if (total_length <= 7660)
                                                return  DISTANCE_NEVER ;
                                            else
                                                return  DISTANCE_MEDIUM ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 12273)
                                        return  DISTANCE_SHORT ;
                                    else
                                        if (total_length <= 12782)
                                        return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 14316)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 14336)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 15345)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 28677)
                            if (total_length <= 26864)
                                if (total_length <= 19436)
                                    if (total_length <= 18160)
                                        if (total_length <= 17392)
                                            if (total_length <= 16113)
                                                return  DISTANCE_MEDIUM ;
                                            else
                                                if (total_length <= 16369)
                                                return  DISTANCE_SHORT ;
                                            else
                                                if (total_length <= 16641)
                                                return  DISTANCE_MEDIUM ;
                                            else
                                                if (total_length <= 16876)
                                                return  DISTANCE_SHORT ;
                                            else
                                                return  DISTANCE_MEDIUM ;
                                        else
                                            if (total_length <= 17393)
                                            return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 18412)
                                        return  DISTANCE_SHORT ;
                                    else
                                        if (total_length <= 19182)
                                        if (total_length <= 18693)
                                            if (total_length <= 18670)
                                                return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 19185)
                                        return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 19200)
                                        return  DISTANCE_NEVER ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    if (total_length <= 23532)
                                    if (total_length <= 22533)
                                        if (total_length <= 19948)
                                            if (total_length <= 19697)
                                                return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            if (total_length <= 21996)
                                            return  DISTANCE_MEDIUM ;
                                        else
                                            if (total_length <= 22001)
                                            return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 22784)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    if (total_length <= 23788)
                                    return  DISTANCE_SHORT ;
                                else
                                    if (total_length <= 24044)
                                    return  DISTANCE_MEDIUM ;
                                else
                                    if (total_length <= 25349)
                                    if (total_length <= 25072)
                                        if (total_length <= 24558)
                                            if (total_length <= 24324)
                                                return  DISTANCE_SHORT ;
                                            else
                                                return  DISTANCE_MEDIUM ;
                                        else
                                            if (total_length <= 24832)
                                            return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 25073)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 26865)
                                return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 28655)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 33772)
                            if (total_length <= 32748)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 32773)
                                return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 33004)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 33029)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 34821)
                            if (total_length <= 34799)
                                if (total_length <= 34565)
                                    if (total_length <= 33776)
                                        return  DISTANCE_SHORT ;
                                    else
                                        if (total_length <= 34540)
                                        return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 35052)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 35056)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 35822)
                            if (total_length <= 35569)
                                if (total_length <= 35565)
                                    return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_NEVER ;
                else
                    if (total_length <= 11249)
                    return  DISTANCE_NEVER ;
                else
                    if (tcp_window <= 64250)
                    if (tcp_window <= 26188)
                        return  DISTANCE_NEVER ;
                    else
                        if (total_length <= 13804)
                        return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_NEVER ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                if (total_length <= 15340)
                if (total_length <= 2798)
                    return  DISTANCE_SHORT ;
                else
                    if (tcp_window <= 2051)
                    if (tcp_window <= 0)
                        return  DISTANCE_LONG ;
                    else
                        if (tcp_window <= 512)
                        if (tcp_window <= 198)
                            if (total_length <= 10221)
                                if (tcp_window <= 17)
                                    return  DISTANCE_NEVER ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 90)
                                if (tcp_window <= 83)
                                    return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (tcp_window <= 296)
                            if (tcp_window <= 259)
                                return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 263)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (tcp_window <= 269)
                                if (tcp_window <= 265)
                                    return  DISTANCE_NEVER ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 10225)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (tcp_window <= 506)
                            if (total_length <= 12805)
                                return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 500)
                                if (tcp_window <= 351)
                                    if (tcp_window <= 338)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 502)
                                return  DISTANCE_NEVER ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (tcp_window <= 511)
                            if (total_length <= 10225)
                                return  DISTANCE_NEVER ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        if (total_length <= 11505)
                        if (total_length <= 5381)
                            if (tcp_window <= 1028)
                                if (tcp_window <= 725)
                                    return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_NEVER ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 10245)
                            if (tcp_window <= 1025)
                                if (tcp_window <= 513)
                                    return  DISTANCE_SHORT ;
                                else
                                    if (tcp_window <= 1023)
                                    if (tcp_window <= 1017)
                                        if (tcp_window <= 515)
                                            return  DISTANCE_NEVER ;
                                        else
                                            if (tcp_window <= 520)
                                            return  DISTANCE_MEDIUM ;
                                        else
                                            return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_NEVER ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 2032)
                                if (tcp_window <= 1062)
                                    if (tcp_window <= 1026)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 2047)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_NEVER ;
                    else
                        if (total_length <= 14318)
                        if (tcp_window <= 513)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (tcp_window <= 1070)
                            if (tcp_window <= 1034)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    if (tcp_window <= 16482)
                    if (tcp_window <= 11309)
                        if (tcp_window <= 2735)
                            if (total_length <= 5381)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 10225)
                                return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 2117)
                                if (tcp_window <= 2079)
                                    return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                if (tcp_window <= 2299)
                                return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 2355)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 5893)
                            return  DISTANCE_SHORT ;
                        else
                            if (tcp_window <= 4099)
                            if (tcp_window <= 4052)
                                return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 10754)
                                return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 4095)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 13297)
                            if (tcp_window <= 8204)
                                if (total_length <= 10757)
                                    if (tcp_window <= 5108)
                                        return  DISTANCE_SHORT ;
                                    else
                                        if (tcp_window <= 5569)
                                        return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 8449)
                                return  DISTANCE_SHORT ;
                            else
                                if (tcp_window <= 8603)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (tcp_window <= 16384)
                        if (tcp_window <= 16382)
                            return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    if (total_length <= 6917)
                    if (tcp_window <= 65410)
                        return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    if (total_length <= 10245)
                    return  DISTANCE_SHORT ;
                else
                    if (tcp_window <= 65450)
                    if (tcp_window <= 64829)
                        return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_NEVER ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (total_length <= 20464)
                if (tcp_window <= 1546)
                    if (total_length <= 16622)
                        return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 17388)
                        return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    if (tcp_window <= 16609)
                    return  DISTANCE_SHORT ;
                else
                    if (tcp_window <= 16811)
                    return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (protocol <= 1)
                return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_SHORT ;
        else
            if (tcp_flags <= 0)
            if (total_length <= 60399)
                if (total_length <= 56305)
                    if (total_length <= 55299)
                        if (total_length <= 40941)
                            if (total_length <= 40689)
                                if (total_length <= 36591)
                                    if (total_length <= 36077)
                                        return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 36100)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    if (total_length <= 36849)
                                    return  DISTANCE_SHORT ;
                                else
                                    if (total_length <= 38404)
                                    if (total_length <= 37615)
                                        if (total_length <= 37357)
                                            return  DISTANCE_MEDIUM ;
                                        else
                                            return  DISTANCE_SHORT ;
                                    else
                                        if (total_length <= 37872)
                                        return  DISTANCE_SHORT ;
                                    else
                                        if (total_length <= 38126)
                                        return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    if (total_length <= 40687)
                                    if (total_length <= 40197)
                                        if (total_length <= 40175)
                                            if (total_length <= 39941)
                                                if (total_length <= 39919)
                                                    if (total_length <= 39664)
                                                        if (total_length <= 39663)
                                                            if (total_length <= 38894)
                                                                return  DISTANCE_MEDIUM ;
                                                            else
                                                                if (total_length <= 38897)
                                                                return  DISTANCE_SHORT ;
                                                            else
                                                                if (total_length <= 39150)
                                                                return  DISTANCE_MEDIUM ;
                                                            else
                                                                if (total_length <= 39173)
                                                                return  DISTANCE_SHORT ;
                                                            else
                                                                return  DISTANCE_MEDIUM ;
                                                        else
                                                            return  DISTANCE_SHORT ;
                                                    else
                                                        return  DISTANCE_MEDIUM ;
                                                else
                                                    return  DISTANCE_SHORT ;
                                            else
                                                return  DISTANCE_MEDIUM ;
                                        else
                                            return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 51185)
                            if (total_length <= 42991)
                                if (total_length <= 42757)
                                    if (total_length <= 42735)
                                        if (total_length <= 42501)
                                            if (total_length <= 42477)
                                                if (total_length <= 42225)
                                                    if (total_length <= 42223)
                                                        if (total_length <= 41968)
                                                            if (total_length <= 41967)
                                                                if (total_length <= 41713)
                                                                    if (total_length <= 41711)
                                                                        if (total_length <= 41201)
                                                                            return  DISTANCE_SHORT ;
                                                                        else
                                                                            return  DISTANCE_MEDIUM ;
                                                                    else
                                                                        return  DISTANCE_SHORT ;
                                                                else
                                                                    return  DISTANCE_MEDIUM ;
                                                            else
                                                                return  DISTANCE_SHORT ;
                                                        else
                                                            return  DISTANCE_MEDIUM ;
                                                    else
                                                        return  DISTANCE_SHORT ;
                                                else
                                                    return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 48113)
                                if (total_length <= 46085)
                                    if (total_length <= 43249)
                                        if (total_length <= 43248)
                                            if (total_length <= 42993)
                                                return  DISTANCE_SHORT ;
                                            else
                                                if (total_length <= 43247)
                                                return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_SHORT ;
                                    else
                                        if (total_length <= 44015)
                                        if (total_length <= 43781)
                                            if (total_length <= 43759)
                                                return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        if (total_length <= 45041)
                                        if (total_length <= 45037)
                                            if (total_length <= 44017)
                                                return  DISTANCE_SHORT ;
                                            else
                                                if (total_length <= 44529)
                                                if (total_length <= 44527)
                                                    return  DISTANCE_MEDIUM ;
                                                else
                                                    return  DISTANCE_SHORT ;
                                            else
                                                return  DISTANCE_MEDIUM ;
                                        else
                                            return  DISTANCE_SHORT ;
                                    else
                                        if (total_length <= 46063)
                                        if (total_length <= 45809)
                                            if (total_length <= 45807)
                                                if (total_length <= 45317)
                                                    return  DISTANCE_SHORT ;
                                                else
                                                    return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    if (total_length <= 47087)
                                    if (total_length <= 46576)
                                        if (total_length <= 46575)
                                            return  DISTANCE_MEDIUM ;
                                        else
                                            return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    if (total_length <= 48111)
                                    if (total_length <= 47857)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 51183)
                                if (total_length <= 49137)
                                    if (total_length <= 49135)
                                        if (total_length <= 48625)
                                            if (total_length <= 48623)
                                                return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    if (total_length <= 50673)
                                    if (total_length <= 50671)
                                        if (total_length <= 50161)
                                            if (total_length <= 50159)
                                                if (total_length <= 49905)
                                                    if (total_length <= 49647)
                                                        return  DISTANCE_MEDIUM ;
                                                    else
                                                        return  DISTANCE_SHORT ;
                                                else
                                                    return  DISTANCE_MEDIUM ;
                                            else
                                                return  DISTANCE_SHORT ;
                                        else
                                            return  DISTANCE_MEDIUM ;
                                    else
                                        return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 52207)
                            if (total_length <= 51953)
                                if (total_length <= 51951)
                                    if (total_length <= 51441)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 52208)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 54252)
                            if (total_length <= 53506)
                                if (total_length <= 52719)
                                    return  DISTANCE_MEDIUM ;
                                else
                                    if (total_length <= 53230)
                                    if (total_length <= 52996)
                                        return  DISTANCE_SHORT ;
                                    else
                                        return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 53742)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 53745)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 54788)
                            if (total_length <= 54767)
                                if (total_length <= 54532)
                                    return  DISTANCE_SHORT ;
                                else
                                    return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 55023)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 55025)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 55279)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 55300)
                        return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 56304)
                        if (total_length <= 55535)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 55537)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 55790)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 56303)
                            if (total_length <= 55793)
                                return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 56045)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 56049)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    if (total_length <= 58605)
                    if (total_length <= 56560)
                        if (total_length <= 56559)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    if (total_length <= 59652)
                    if (total_length <= 59631)
                        if (total_length <= 58608)
                            return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                if (total_length <= 60400)
                return  DISTANCE_SHORT ;
            else
                if (total_length <= 63491)
                if (total_length <= 62956)
                    return  DISTANCE_MEDIUM ;
                else
                    if (total_length <= 63216)
                    return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                if (total_length <= 65007)
                if (total_length <= 63492)
                    return  DISTANCE_SHORT ;
                else
                    if (total_length <= 63984)
                    return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                if (total_length <= 65008)
                return  DISTANCE_SHORT ;
            else
                if (total_length <= 65519)
                return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_SHORT ;
        else
            if (total_length <= 49137)
            if (tcp_window <= 252)
                if (tcp_window <= 80)
                    if (tcp_window <= 65)
                        return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (tcp_window <= 505)
                return  DISTANCE_SHORT ;
            else
                if (tcp_window <= 515)
                if (total_length <= 43761)
                    return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_SHORT ;
        else
            return  DISTANCE_SHORT ;
    else
        if (tcp_flags <= 20)
        if (tcp_window <= 24083)
            if (total_length <= 10225)
                if (tcp_flags <= 17)
                    if (tcp_window <= 1420)
                        if (tcp_window <= 483)
                            return  DISTANCE_NEVER ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_NEVER ;
                else
                    return  DISTANCE_NEVER ;
            else
                if (tcp_window <= 2006)
                if (tcp_window <= 247)
                    return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_NEVER ;
            else
                return  DISTANCE_MEDIUM ;
        else
            if (tcp_window <= 65254)
            if (tcp_window <= 63999)
                return  DISTANCE_MEDIUM ;
            else
                if (tcp_window <= 64250)
                return  DISTANCE_NEVER ;
            else
                return  DISTANCE_MEDIUM ;
        else
            return  DISTANCE_MEDIUM ;
    else
        if (total_length <= 30703)
        if (total_length <= 17388)
            if (tcp_window <= 500)
                if (total_length <= 16899)
                    return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (tcp_window <= 2531)
                if (tcp_window <= 2399)
                    if (tcp_window <= 925)
                        if (total_length <= 496)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 2797)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 4590)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 9477)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (tcp_window <= 608)
                            if (tcp_window <= 506)
                                return  DISTANCE_SHORT ;
                            else
                                if (total_length <= 13317)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 16877)
                        if (total_length <= 2545)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 15598)
                            if (tcp_window <= 2049)
                                if (tcp_window <= 1572)
                                    return  DISTANCE_MEDIUM ;
                                else
                                    return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (tcp_flags <= 24)
                if (tcp_window <= 65393)
                    if (total_length <= 16369)
                        if (total_length <= 15341)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                return  DISTANCE_MEDIUM ;
        else
            if (tcp_flags <= 24)
            if (total_length <= 17649)
                if (tcp_window <= 1025)
                    return  DISTANCE_LONG ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (tcp_window <= 176)
                if (total_length <= 20465)
                    if (total_length <= 20209)
                        if (total_length <= 18156)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (tcp_window <= 146)
                        if (tcp_window <= 101)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    if (total_length <= 27909)
                    if (total_length <= 25601)
                        if (tcp_window <= 113)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (total_length <= 22764)
                if (total_length <= 20994)
                    if (total_length <= 18414)
                        if (total_length <= 18181)
                            if (total_length <= 17925)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        if (total_length <= 19436)
                        if (tcp_window <= 508)
                            return  DISTANCE_SHORT ;
                        else
                            if (total_length <= 18924)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (tcp_window <= 1893)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 20224)
                        return  DISTANCE_MEDIUM ;
                    else
                        if (tcp_window <= 2052)
                        return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    if (total_length <= 22513)
                    if (tcp_window <= 2042)
                        return  DISTANCE_MEDIUM ;
                    else
                        if (tcp_window <= 2053)
                        return  DISTANCE_NEVER ;
                    else
                        if (total_length <= 21232)
                        return  DISTANCE_NEVER ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_NEVER ;
            else
                if (total_length <= 24045)
                return  DISTANCE_MEDIUM ;
            else
                if (total_length <= 24300)
                return  DISTANCE_SHORT ;
            else
                if (tcp_window <= 63750)
                if (total_length <= 28653)
                    return  DISTANCE_MEDIUM ;
                else
                    if (total_length <= 30193)
                    return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                return  DISTANCE_SHORT ;
        else
            return  DISTANCE_NEVER ;
    else
        if (total_length <= 30725)
        return  DISTANCE_MEDIUM ;
    else
        if (tcp_window <= 500)
        if (tcp_window <= 119)
            if (tcp_window <= 82)
                if (total_length <= 56304)
                    if (tcp_window <= 26)
                        return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                if (tcp_window <= 83)
                return  DISTANCE_SHORT ;
            else
                if (tcp_window <= 84)
                return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_SHORT ;
        else
            if (tcp_flags <= 24)
            if (total_length <= 35824)
                if (tcp_window <= 205)
                    return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                if (total_length <= 56323)
                if (tcp_window <= 263)
                    if (tcp_window <= 255)
                        return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_MEDIUM ;
        else
            return  DISTANCE_NEVER ;
    else
        if (total_length <= 65520)
        if (tcp_window <= 1021)
            if (tcp_window <= 1020)
                if (tcp_window <= 516)
                    if (total_length <= 35845)
                        return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 37361)
                        if (tcp_window <= 501)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 54255)
                        if (tcp_window <= 507)
                            if (total_length <= 43757)
                                return  DISTANCE_MEDIUM ;
                            else
                                if (total_length <= 46085)
                                return  DISTANCE_SHORT ;
                            else
                                return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        if (total_length <= 56324)
                        if (tcp_window <= 501)
                            return  DISTANCE_MEDIUM ;
                        else
                            return  DISTANCE_SHORT ;
                    else
                        if (total_length <= 61422)
                        return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_SHORT ;
        else
            if (tcp_window <= 3483)
            if (tcp_window <= 2115)
                if (tcp_window <= 2049)
                    if (total_length <= 43756)
                        if (tcp_window <= 1040)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (total_length <= 35589)
                            if (total_length <= 32748)
                                return  DISTANCE_MEDIUM ;
                            else
                                return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        if (total_length <= 49412)
                        return  DISTANCE_SHORT ;
                    else
                        return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                if (tcp_window <= 2255)
                return  DISTANCE_MEDIUM ;
            else
                if (tcp_window <= 2657)
                if (total_length <= 36079)
                    return  DISTANCE_SHORT ;
                else
                    if (tcp_window <= 2438)
                    return  DISTANCE_SHORT ;
                else
                    return  DISTANCE_MEDIUM ;
            else
                return  DISTANCE_MEDIUM ;
        else
            if (total_length <= 56303)
            if (total_length <= 53745)
                if (tcp_window <= 65534)
                    if (total_length <= 49389)
                        if (tcp_window <= 15309)
                            return  DISTANCE_MEDIUM ;
                        else
                            if (tcp_window <= 22162)
                            return  DISTANCE_SHORT ;
                        else
                            return  DISTANCE_MEDIUM ;
                    else
                        return  DISTANCE_SHORT ;
                else
                    if (total_length <= 46316)
                    return  DISTANCE_MEDIUM ;
                else
                    return  DISTANCE_SHORT ;
            else
                if (total_length <= 55276)
                return  DISTANCE_NEVER ;
            else
                return  DISTANCE_MEDIUM ;
        else
            if (total_length <= 56305)
            return  DISTANCE_SHORT ;
        else
            if (total_length <= 62468)
            if (total_length <= 59374)
                return  DISTANCE_SHORT ;
            else
                return  DISTANCE_MEDIUM ;
        else
            return  DISTANCE_SHORT ;
    else
        return  DISTANCE_MEDIUM ;
}
} // namespace ipxp