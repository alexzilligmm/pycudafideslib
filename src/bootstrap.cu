// bootstrap.cu
// Bootstrapping via FIDESlib's GPU-accelerated EvalBootstrap().
//
// FIDESlib implements the full CKKS bootstrapping circuit (Han & Ki 2020):
//   ModRaise → CoeffToSlot → EvalMod (composite sin) → SlotToCoeff
// All stages run on GPU.  We simply delegate to cc->EvalBootstrap().
//
// The setup (EvalBootstrapSetup + EvalBootstrapKeyGen) is done once in
// make_ckks_context() inside fideslib_wrapper.h.

#include "llama.h"
#include <iostream>

// ── public API (used by nonlinear.cu and llama.cu) ────────────────────────

// Bootstrap ct and drop it to `target_level` depth from fresh.
// target_level == 0 means use ct at maximum freshness.
Ctx bootstrap_to(const LlamaInference& llama, const Ctx& ct,
                  uint32_t target_level) {
    const CC& cc = llama.cc();
    std::cout << "Bootstrapping (level " << level_of(ct)
              << " → ~0, then drop to " << target_level << ")...\n";

    Timer t;
    Ctx fresh = cc->EvalBootstrap(ct);   // FIDESlib GPU-accelerated circuit

    // Consume levels until we reach the desired starting point.
    drop_levels(cc, fresh, target_level);

    std::cout << "  bootstrap done in " << t.elapsed_s()
              << " s, output level " << level_of(fresh) << "\n";
    return fresh;
}
