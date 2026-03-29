#include "inference.h"
#include <iostream>

Ctx bootstrap_to(Inference& inf, const Ctx& ct,
                  uint32_t target_remaining) {
    const CC& cc = inf.cc();
    uint32_t total    = (uint32_t)inf.total_depth;
    uint32_t consumed = level_of(ct);
    uint32_t remaining = (total > consumed) ? (total - consumed) : 0u;

    if (remaining >= target_remaining) {
        std::cout << "bootstrap_to(" << target_remaining
                  << "): already at " << remaining << " remaining, skip\n";
        return ct;
    }

    Timer t;
    Ctx fresh = cc->EvalBootstrap(ct);

    if (target_remaining > 0) {
        uint32_t fresh_con  = level_of(fresh);
        uint32_t want_con   = (total > target_remaining)
                              ? (total - target_remaining) : 0u;
        if (fresh_con < want_con)
            drop_levels(cc, fresh, want_con - fresh_con);
    }

    return fresh;
}
