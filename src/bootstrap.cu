#include "inference.h"
#include <iostream>

Ctx bootstrap_to(Inference& inf, const Ctx& ct,
                  uint32_t target_remaining) {
    const CC& cc = inf.cc();
    std::cout << "Bootstrapping (level " << level_of(ct) << ")...\n";

    Timer t;
    Ctx fresh = cc->EvalBootstrap(ct);   // FIDESlib GPU-accelerated circuit

    std::cout << "  bootstrap done in " << t.elapsed_s()
              << " s, raw output level " << level_of(fresh) << "\n";

    if (target_remaining > 0) {
        uint32_t total      = (uint32_t)inf.total_depth;
        uint32_t fresh_con  = level_of(fresh);               // consumed now
        uint32_t want_con   = (total > target_remaining)
                              ? (total - target_remaining) : 0u;
        if (fresh_con < want_con)
            drop_levels(cc, fresh, want_con - fresh_con);
    }

    std::cout << "  output level after target drop: " << level_of(fresh) << "\n";
    return fresh;
}
