#pragma once
#include <string>
#include "model/expert.h"

namespace io {

static constexpr uint32_t CKPT_MAGIC   = 0x45585054; // "EXPT"
static constexpr uint32_t CKPT_VERSION = 1;

// Load-only at inference time.
void checkpoint_load(const std::string& path, model::ExpertModel& e);

} // namespace io