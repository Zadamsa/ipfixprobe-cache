#include "packetclassifier.hpp"

namespace ipxp {
//Prijima atributy paketu. Navratova hodnota je vzdalenost dalsiho paketu v tom samem flow
PacketDistance PacketClassifier::classifyInstance(uint8_t tcp_flags, uint16_t tcp_window, uint16_t total_length) {
    if (tcp_flags <= 16) {
        if (tcp_flags <= 4) {
            if (total_length <= 11776) {
                return PacketDistance::DISTANCE_NEVER;
            } else {
                return PacketDistance::DISTANCE_MEDIUM;
            }
        } else {
            if (total_length <= 19973) {
                if (total_length <= 2821) {
                    return PacketDistance::DISTANCE_SHORT;
                } else {
                    if (tcp_window <= 2122) {
                        if (tcp_window <= 512) {
                            if (tcp_window <= 500) {
                                if (tcp_window <= 259) {
                                    return PacketDistance::DISTANCE_SHORT;
                                } else {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                }
                            } else {
                                return PacketDistance::DISTANCE_MEDIUM;
                            }
                        } else {
                            if (tcp_window <= 520) {
                                if (total_length <= 10245) {
                                    return PacketDistance::DISTANCE_SHORT;
                                } else {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                }
                            } else {
                                return PacketDistance::DISTANCE_SHORT;
                            }
                        }
                    } else {
                        if (tcp_window <= 16482) {
                            return PacketDistance::DISTANCE_SHORT;
                        } else {
                            if (tcp_window <= 17522) {
                                return PacketDistance::DISTANCE_MEDIUM;
                            } else {
                                return PacketDistance::DISTANCE_SHORT;
                            }
                        }
                    }
                }
            } else {
                return PacketDistance::DISTANCE_SHORT;
            }
        }
    } else {
        if (tcp_flags <= 20) {
            if (total_length <= 10752) {
                return PacketDistance::DISTANCE_NEVER;
            } else {
                if (tcp_window <= 500) {
                    if (total_length <= 35844) {
                        if (total_length <= 2565) {
                            if (tcp_window <= 220) {
                                if (tcp_window <= 66) {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                } else {
                                    if (total_length <= 20483) {
                                        return PacketDistance::DISTANCE_SHORT;
                                    } else {
                                        return PacketDistance::DISTANCE_MEDIUM;
                                    }
                                }
                            } else {
                                return PacketDistance::DISTANCE_MEDIUM;
                            }
                        } else {
                            return PacketDistance::DISTANCE_MEDIUM;
                        }
                    } else {
                        if (total_length <= 17408) {
                            if (tcp_window <= 2533) {
                                if (total_length <= 11525) {
                                    return PacketDistance::DISTANCE_SHORT;
                                } else {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                }
                            } else {
                                return PacketDistance::DISTANCE_SHORT;
                            }
                        } else {
                            if (total_length <= 25093) {
                                if (total_length <= 23809) {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                } else {
                                    return PacketDistance::DISTANCE_SHORT;
                                }
                            } else {
                                if (total_length <= 29443) {
                                    if (total_length <= 22784) {
                                        if (total_length <= 20997) {
                                            if (total_length <= 18432) {
                                                return PacketDistance::DISTANCE_MEDIUM;
                                            } else {
                                                return PacketDistance::DISTANCE_SHORT;
                                            }
                                        } else {
                                            return PacketDistance::DISTANCE_MEDIUM;
                                        }
                                    } else {
                                        return PacketDistance::DISTANCE_SHORT;
                                    }
                                } else {
                                    if (total_length <= 43013) {
                                        if (total_length <= 43012) {
                                            if (tcp_window <= 501) {
                                                if (total_length <= 36608) {
                                                    return PacketDistance::DISTANCE_SHORT;
                                                } else {
                                                    return PacketDistance::DISTANCE_MEDIUM;
                                                }
                                            } else {
                                                return PacketDistance::DISTANCE_MEDIUM;
                                            }
                                        } else {
                                            return PacketDistance::DISTANCE_MEDIUM;
                                        }
                                    } else {
                                        if (total_length <= 43264) {
                                            if (tcp_window <= 3437) {
                                                if (total_length <= 56321) {
                                                    return PacketDistance::DISTANCE_MEDIUM;
                                                } else {
                                                    return PacketDistance::DISTANCE_SHORT;
                                                }
                                            } else {
                                                return PacketDistance::DISTANCE_MEDIUM;
                                            }
                                        } else {
                                            return PacketDistance::DISTANCE_SHORT;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    return PacketDistance::DISTANCE_MEDIUM;
                }
            }
        } else {
            if (tcp_flags <= 30) {
                if (tcp_window <= 500) {
                    if (total_length <= 35844) {
                        if (total_length <= 32771) {
                            return PacketDistance::DISTANCE_MEDIUM;
                        } else {
                            return PacketDistance::DISTANCE_SHORT;
                        }
                    } else {
                        return PacketDistance::DISTANCE_MEDIUM;
                    }
                } else {
                    return PacketDistance::DISTANCE_MEDIUM;
                }
            } else {
                return PacketDistance::DISTANCE_MEDIUM;
            }
        }
    }
}
} // namespace ipxp