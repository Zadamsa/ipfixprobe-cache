//
// Created by zaida on 13.02.2024.
//

#include "packetclassifier.hpp"

namespace ipxp {
PacketDistance PacketClassifier::classifyInstance(uint8_t attribute0, uint16_t attribute1, uint16_t attribute2) {
    if (attribute0 <= 16) {
        if (attribute0 <= 4) {
            if (attribute2 <= 11776) {
                return PacketDistance::DISTANCE_NEVER;
            } else {
                return PacketDistance::DISTANCE_MEDIUM;
            }
        } else {
            if (attribute2 <= 19973) {
                if (attribute2 <= 2821) {
                    return PacketDistance::DISTANCE_SHORT;
                } else {
                    if (attribute1 <= 2122) {
                        if (attribute1 <= 512) {
                            if (attribute1 <= 500) {
                                if (attribute1 <= 259) {
                                    return PacketDistance::DISTANCE_SHORT;
                                } else {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                }
                            } else {
                                return PacketDistance::DISTANCE_MEDIUM;
                            }
                        } else {
                            if (attribute1 <= 520) {
                                if (attribute2 <= 10245) {
                                    return PacketDistance::DISTANCE_SHORT;
                                } else {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                }
                            } else {
                                return PacketDistance::DISTANCE_SHORT;
                            }
                        }
                    } else {
                        if (attribute1 <= 16482) {
                            return PacketDistance::DISTANCE_SHORT;
                        } else {
                            if (attribute1 <= 17522) {
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
        if (attribute0 <= 20) {
            if (attribute2 <= 10752) {
                return PacketDistance::DISTANCE_NEVER;
            } else {
                if (attribute1 <= 500) {
                    if (attribute2 <= 35844) {
                        if (attribute2 <= 2565) {
                            if (attribute1 <= 220) {
                                if (attribute1 <= 66) {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                } else {
                                    if (attribute2 <= 20483) {
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
                        if (attribute2 <= 17408) {
                            if (attribute1 <= 2533) {
                                if (attribute2 <= 11525) {
                                    return PacketDistance::DISTANCE_SHORT;
                                } else {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                }
                            } else {
                                return PacketDistance::DISTANCE_SHORT;
                            }
                        } else {
                            if (attribute2 <= 25093) {
                                if (attribute2 <= 23809) {
                                    return PacketDistance::DISTANCE_MEDIUM;
                                } else {
                                    return PacketDistance::DISTANCE_SHORT;
                                }
                            } else {
                                if (attribute2 <= 29443) {
                                    if (attribute2 <= 22784) {
                                        if (attribute2 <= 20997) {
                                            if (attribute2 <= 18432) {
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
                                    if (attribute2 <= 43013) {
                                        if (attribute2 <= 43012) {
                                            if (attribute1 <= 501) {
                                                if (attribute2 <= 36608) {
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
                                        if (attribute2 <= 43264) {
                                            if (attribute1 <= 3437) {
                                                if (attribute2 <= 56321) {
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
            if (attribute0 <= 30) {
                if (attribute1 <= 500) {
                    if (attribute2 <= 35844) {
                        if (attribute2 <= 32771) {
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