#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <future>
#include <ctt_card.hpp>
#include <ipfixprobe/cttmeta.hpp>

namespace ipxp {

std::shared_ptr<ctt::AsyncCommander<KEY_SIZE, STATE_SIZE, MASK_SIZE>> get_ctt_commander(const std::string& nfb_dev, unsigned ctt_comp_index) noexcept;


} // namespace ipxp