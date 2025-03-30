/**
* \file
 * \author Damir Zainullin <zaidamilda@gmail.com>
 * \brief CttController declaration.
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

#pragma once

#define DISABLE_CODE 1
#if DISABLE_CODE
    #define MAYBE_DISABLED_CODE(...)
#else
    #define MAYBE_DISABLED_CODE(...) do { __VA_ARGS__; } while (0);
#endif

#include <config.h>
#ifdef WITH_CTT
#include <ipfixprobe/cttmeta.hpp>
#include <ipfixprobe/flowifc.hpp>
#include <sys/time.h>
#include <ctt_async.hpp>
#include <ctt_factory.hpp>
#include <ctt_exceptions.hpp>
#include <ctt_modes.hpp>
#include <ctt.hpp>
#include <queue>
#include <tuple>
#include <cstddef>
#include <vector>

namespace ipxp {

class CttController {
public:
   /**
   * @brief init the CTT.
   *
   * @param nfb_dev          The NFB device file (e.g., "/dev/nfb0").
   * @param ctt_comp_index   The index of the CTT component.
   */
   CttController(const std::string& nfb_dev, unsigned ctt_comp_index);

   /**
   * @brief Command: mark a flow for offload.
   *
   * @param flow_hash_ctt    The flow hash to be offloaded.
   */
   void create_record(const Flow& flow, uint8_t dma_channel, feta::OffloadMode offload_mode);

   /**
   * @brief Command: export a flow from the CTT.
   *
   * @param flow_hash_ctt    The flow hash to be exported.
   */
   void export_record(uint64_t flow_hash_ctt);

   ~CttController() noexcept;

   void remove_record_without_notification(uint64_t flow_hash_ctt);

   void get_state(uint64_t flow_hash_ctt);
private:
   std::unique_ptr<ctt::AsyncCommander> m_commander;
   size_t m_key_size_bytes;
   size_t m_state_size_bytes;
   size_t m_state_mask_size_bytes;

   /**
   * @brief Assembles the state vector from the given values.
   *
   * @param offload_mode     The offload mode.
   * @param meta_type        The metadata type.
   * @param timestamp_first  The first timestamp of the flow.
   * @return A byte vector representing the assembled state vector.
   */
   std::vector<std::byte>
   assemble_state(feta::OffloadMode offload_mode, feta::MetaType meta_type, const Flow& flow, uint8_t dma_channel);

   /**
   * @brief Assembles the key vector from the given flow hash.
   *
   * @param flow_hash_ctt    The flow hash.
   * @return A byte vector representing the assembled key vector.
   */
   std::vector<std::byte> assemble_key(uint64_t flow_hash_ctt);

   std::pair<std::vector<std::byte>, std::vector<std::byte>>
   get_key_and_state(uint64_t flow_hash_ctt, const Flow& flow, uint8_t dma_channel);
};

} // ipxp

#endif /* WITH_CTT */
