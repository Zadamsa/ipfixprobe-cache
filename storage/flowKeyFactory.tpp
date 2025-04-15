#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <variant>
#include <optional>
#include "flowKey.hpp"

namespace ipxp {

class FlowKeyFactory {
public:
   template<typename Int>
   static FlowKey
   create_direct_key(const Int* src_ip, const Int* dst_ip,
      uint16_t src_port, uint16_t dst_port, uint8_t proto, IP ip_version, uint16_t vlan_id) noexcept
   {
      FlowKey res;
      if (ip_version == IP::v4) {   
         *reinterpret_cast<uint64_t*>(&res.src_ip[0]) = 0;
         *reinterpret_cast<uint32_t*>(&res.src_ip[8]) = htobe32(0x0000FFFF);
         *reinterpret_cast<uint32_t*>(&res.src_ip[12]) = *reinterpret_cast<const uint32_t*>(src_ip);
         *reinterpret_cast<uint64_t*>(&res.dst_ip[0]) = 0;
         *reinterpret_cast<uint32_t*>(&res.dst_ip[8]) = htobe32(0x0000FFFF);
         *reinterpret_cast<uint32_t*>(&res.dst_ip[12]) = *reinterpret_cast<const uint32_t*>(dst_ip);
      } else if (ip_version == IP::v6) {
         std::memcpy(res.src_ip.begin(), src_ip, 16);
         std::memcpy(res.dst_ip.begin(), dst_ip, 16);
      }
      res.src_port = src_port;
      res.dst_port = dst_port;
      res.proto = proto;
      res.ip_version = ip_version;
      res.vlan_id = vlan_id;
      return res;
   }

   template<typename Int>
   static FlowKey
   create_reversed_key(const Int* src_ip, const Int* dst_ip,
      uint16_t src_port, uint16_t dst_port, uint8_t proto, IP ip_version, uint16_t vlan_id) noexcept
   {
      FlowKey res;
      if (ip_version == IP::v4) {   
         *reinterpret_cast<uint64_t*>(&res.dst_ip[0]) = 0;
         *reinterpret_cast<uint32_t*>(&res.dst_ip[8]) = htobe32(0x0000FFFF);
         *reinterpret_cast<uint32_t*>(&res.dst_ip[12]) = *reinterpret_cast<const uint32_t*>(src_ip);
         *reinterpret_cast<uint64_t*>(&res.src_ip[0]) = 0;
         *reinterpret_cast<uint32_t*>(&res.src_ip[8]) = htobe32(0x0000FFFF);
         *reinterpret_cast<uint32_t*>(&res.src_ip[12]) = *reinterpret_cast<const uint32_t*>(dst_ip);
      } else if (ip_version == IP::v6) {
         std::memcpy(res.src_ip.begin(), dst_ip, 16);
         std::memcpy(res.dst_ip.begin(), src_ip, 16);
      }
      res.src_port = dst_port;
      res.dst_port = src_port;
      res.proto = proto;
      res.ip_version = ip_version;
      res.vlan_id = vlan_id;
      return res;
   }

   template<typename Int>
   static std::pair<FlowKey, bool>
   create_sorted_key(const Int* src_ip, const Int* dst_ip,
      uint16_t src_port, uint16_t dst_port, uint8_t proto, IP ip_version, uint16_t vlan_id) noexcept
   {
      if (src_port <= dst_port) {
         return {create_direct_key(src_ip, dst_ip, src_port, dst_port, proto, ip_version, vlan_id), false};
      }
      return {create_reversed_key(src_ip, dst_ip, src_port, dst_port, proto, ip_version, vlan_id), true};
   }
};

} // ipxp