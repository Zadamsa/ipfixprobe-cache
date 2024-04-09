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
#include <tuple>
#include "timevalUtils.hpp"

namespace ipxp {

struct timeval operator+(const struct timeval& a, const struct timeval& b) noexcept
{
   constexpr time_t USEC_IN_SEC = 1000000;

   struct timeval result;
   result.tv_sec = a.tv_sec + b.tv_sec;
   result.tv_usec = a.tv_usec + b.tv_usec;
   if (result.tv_usec >= USEC_IN_SEC) {
       result.tv_sec++;
       result.tv_usec -= USEC_IN_SEC;
   }
   return result;
}

struct timeval operator-(const struct timeval& a, const struct timeval& b) noexcept
{
   constexpr time_t USEC_IN_SEC = 1000000;

   struct timeval result;
   result.tv_sec = a.tv_sec - b.tv_sec;
   if (a.tv_usec >= b.tv_usec)
       result.tv_usec = a.tv_usec - b.tv_usec;
   else{
       result.tv_usec = USEC_IN_SEC + a.tv_usec - b.tv_usec;
       result.tv_sec--;
   }
   return result;
}

bool operator>(const struct timeval& a, const struct timeval& b) noexcept
{
   if (a.tv_sec == b.tv_sec)
       return a.tv_usec > b.tv_usec;
   return a.tv_sec > b.tv_sec;
}

bool operator==(const struct timeval& a, const struct timeval& b) noexcept
{
   return std::tie(a.tv_sec,a.tv_usec) == std::tie(b.tv_sec,b.tv_usec);
}

bool operator!=(const struct timeval& a, const struct timeval& b) noexcept
{
   return !(a==b);
}

bool operator<(const struct timeval& a, const struct timeval& b) noexcept
{
   return !(a>b || a==b);
}

bool operator>=(const struct timeval& a, const struct timeval& b) noexcept
{
   return a>b || a==b;
}



} // namespace ipxp
