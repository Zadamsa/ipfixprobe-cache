/**
 * \file
 * \author Pavel Siska <siska@cesnet.cz>
 * \brief Utils for timeval struct
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

#include <sys/time.h>

#pragma once

namespace ipxp {

struct timeval operator+(const struct timeval& a, const struct timeval& b) noexcept;

struct timeval operator-(const struct timeval& a, const struct timeval& b) noexcept;

bool operator>(const struct timeval& a, const struct timeval& b) noexcept;

bool operator==(const struct timeval& a, const struct timeval& b) noexcept;

bool operator!=(const struct timeval& a, const struct timeval& b) noexcept;

bool operator<(const struct timeval& a, const struct timeval& b) noexcept;

bool operator>=(const struct timeval& a, const struct timeval& b) noexcept;

} // namespace ipxp
