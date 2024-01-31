/**
* \file flowendreason.hpp
* \brief Reasons of exporting cache
 */
/*
* Copyright (C) 2023 CESNET
*
* LICENSE TERMS
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in
*    the documentation and/or other materials provided with the
*    distribution.
* 3. Neither the name of the Company nor the names of its contributors
*    may be used to endorse or promote products derived from this
*    software without specific prior written permission.
*/



#ifndef IPFIXPROBE_CACHE_FLOWENDREASON_HPP
#define IPFIXPROBE_CACHE_FLOWENDREASON_HPP

namespace ipxp {
enum FlowEndReason:uint8_t {
    FLOW_END_INACTIVE_TIMEOUT,
    FLOW_END_ACTIVE_TIMEOUT,
    FLOW_END_TCP_EOF,
    FLOW_END_CACHE_SHUTDOWN,
    FLOW_END_POST_UPDATE,
    FLOW_END_PRE_UPDATE,
    FLOW_END_POST_CREATE,
    FLOW_END_NO_ROW_SPACE
};

} // namespace ipxp

#endif // IPFIXPROBE_CACHE_FLOWENDREASON_HPP
