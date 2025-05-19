#include "asyncCommanderFactory.hpp"

namespace ipxp {

std::unique_ptr<ctt::Card<KEY_SIZE, STATE_SIZE, MASK_SIZE>> g_card;

std::shared_ptr<ctt::AsyncCommander<KEY_SIZE, STATE_SIZE, MASK_SIZE>> get_ctt_commander(const std::string& nfb_dev, unsigned ctt_comp_index) noexcept
{
    if (!g_card) {
        g_card = std::make_unique<ctt::Card<KEY_SIZE, STATE_SIZE, MASK_SIZE>>(nfb_dev);
    }
    return g_card->get_async_commander(ctt_comp_index);
}



} // namespace ipxp