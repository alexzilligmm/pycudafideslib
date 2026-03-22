#pragma once
// fideslib_wrapper.h
// Thin convenience layer on top of FIDESlib (OpenFHE 1.4.2 + GPU acceleration).
// Every other file in this project #includes only this header for FHE types.
//
// FIDESlib API lives in the fideslib:: namespace (not lbcrypto::).
// Key differences vs raw OpenFHE:
//   • cc->LoadContext(pk)  — requires the public key argument
//   • EvalBootstrapSetup takes 4 args (adds correctionFactor)
//   • Decrypt takes (Ctx&, const SK&, Plaintext*)  — ct is non-const ref
//   • EvalAddInPlace / EvalMultInPlace take Plaintext& (non-const) — no temporaries
//   • No LevelReduceInPlace — use CryptoContextImpl<DCRTPoly>::SetLevel instead

#include <fideslib.hpp>   // brings in all fideslib headers (CCParams, CryptoContext, etc.)

#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <stdexcept>
#include <memory>

// ── type aliases ──────────────────────────────────────────────────────────
using namespace fideslib;

using CC   = CryptoContext<DCRTPoly>;            // shared_ptr<CryptoContextImpl<DCRTPoly>>
using Ctx  = Ciphertext<DCRTPoly>;               // shared_ptr<CiphertextImpl<DCRTPoly>>
using Ptx  = Plaintext;                          // shared_ptr<PlaintextImpl>
using KP   = KeyPair<DCRTPoly>;

// ── CKKS context bundle ───────────────────────────────────────────────────
struct CKKSContext {
    CC  cc;
    KP  keys;

    PublicKey<DCRTPoly>&  pk()  { return keys.publicKey; }
    PrivateKey<DCRTPoly>& sk()  { return keys.secretKey; }
};

// ── parameter factory ────────────────────────────────────────────────────
inline std::shared_ptr<CKKSContext>
make_ckks_context(int logN                  = 12,
                  int depth                 = 16,
                  int scale_bits            = 40,
                  uint32_t bootstrap_slots  = 0,    // 0 = N/2
                  bool enable_bootstrap     = true) {
    CCParams<CryptoContextCKKSRNS> params;

    // Bootstrap requires higher precision: use 59-bit scaling as in the
    // FIDESlib reference example (dcrtBits=59, firstMod=60, depth=25,
    // levelBudget={3,3}).  Non-bootstrap contexts keep user-supplied scale_bits.
    std::vector<uint32_t> level_budget;
    uint32_t approx_bootstrap_depth;
    int actual_scale_bits;
    int first_mod;
    if (enable_bootstrap) {
        actual_scale_bits       = 59;
        first_mod               = 60;
        level_budget            = {3, 3};
        approx_bootstrap_depth  = 9;  // total depth = depth + 9 = 25 (matching FIDESlib example)
    } else {
        actual_scale_bits       = scale_bits;
        first_mod               = 60;
        level_budget            = {};
        approx_bootstrap_depth  = 0;
    }

    params.SetMultiplicativeDepth(depth + approx_bootstrap_depth);
    params.SetScalingModSize(actual_scale_bits);
    params.SetFirstModSize(first_mod);
    params.SetScalingTechnique(FLEXIBLEAUTO);
    params.SetBatchSize(1 << (logN - 1));
    params.SetSecretKeyDist(UNIFORM_TERNARY);
    params.SetNumLargeDigits(3);
    params.SetKeySwitchTechnique(HYBRID);
    if (enable_bootstrap) {
        params.SetSecurityLevel(HEStd_NotSet);  // required for bootstrap ring-dim
        params.SetRingDim(1 << logN);           // must be explicit when HEStd_NotSet
    }
    // Note: SetLevelBudget / SetBsgsDim do not exist in fideslib::CCParams;
    // bootstrapping layout is configured via EvalBootstrapSetup below.

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    if (enable_bootstrap) {
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
    }

    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    // ── rotation key generation ───────────────────────────────────────────
    // Covers:
    //   Powers of 2 (1..slots/2): inner BSGS steps for all power-of-2 hidDim.
    //   Negatives: RoPE xsin1 uses -intRot = -(slots / hidDim).
    //   1024/j fractions: softmax/argmax sum-over-heads rotations.
    //   Legacy 5/-5: retained for compatibility (was Go test key).
    //
    // For large hidDim (e.g., 4096) outer BSGS steps may not be powers of 2.
    // In that case the caller must extend this list via make_ckks_context_with_steps().
    const int max_slots = 1 << (logN - 1);
    std::vector<int32_t> rot_steps;
    for (int i = 1; i <= max_slots; i *= 2) rot_steps.push_back(i);
    for (int i = 1; i <= max_slots; i *= 2) rot_steps.push_back(-i);
    for (int j = 1; j <= 256; j *= 2) {
        rot_steps.push_back(1024 / j);
        rot_steps.push_back(-(1024 / j));
    }
    rot_steps.push_back(5);
    rot_steps.push_back(-5);
    std::sort(rot_steps.begin(), rot_steps.end());
    rot_steps.erase(std::unique(rot_steps.begin(), rot_steps.end()), rot_steps.end());
    cc->EvalRotateKeyGen(kp.secretKey, rot_steps);

    if (enable_bootstrap) {
        uint32_t slots = (bootstrap_slots == 0) ? (1u << (logN - 1)) : bootstrap_slots;
        // fideslib EvalBootstrapSetup: (levelBudget, dim1, slots, correctionFactor)
        cc->EvalBootstrapSetup(level_budget, {0, 0}, slots, /*correctionFactor=*/0);
        cc->EvalBootstrapKeyGen(kp.secretKey, slots);
    }

    // Load the context (parameters, eval keys, NTT tables) to GPU.
    // fideslib requires the public key to be passed here.
    cc->LoadContext(kp.publicKey);

    auto ctx  = std::make_shared<CKKSContext>();
    ctx->cc   = cc;
    ctx->keys = std::move(kp);
    return ctx;
}

// ── encode / encrypt helpers ─────────────────────────────────────────────

inline Ptx encode(const CC& cc, const std::vector<double>& values, int level = 0) {
    return cc->MakeCKKSPackedPlaintext(values, /*noiseScaleDeg=*/1, (uint32_t)level);
}

inline Ptx encode(const CC& cc, const std::vector<std::complex<double>>& values,
                   int level = 0) {
    return cc->MakeCKKSPackedPlaintext(values, /*noiseScaleDeg=*/1, (uint32_t)level);
}

inline Ptx encode_const(const CC& cc, double val, size_t slots, int level = 0) {
    std::vector<double> v(slots, val);
    return encode(cc, v, level);
}

// Encrypt: fideslib Encrypt takes Plaintext& (non-const)
inline Ctx encrypt(const CC& cc, Ptx pt, const PublicKey<DCRTPoly>& pk) {
    return cc->Encrypt(pk, pt);
}

inline Ctx encrypt_const(const CC& cc, double val, size_t slots,
                          const PublicKey<DCRTPoly>& pk, int level = 0) {
    return encrypt(cc, encode_const(cc, val, slots, level), pk);
}

// Decrypt: fideslib signature is Decrypt(Ctx&, const SK&, Plaintext*)
// ct must be non-const; take by value (copy of shared_ptr) to allow non-const bind.
inline std::vector<double> decrypt(const CC& cc, Ctx ct,
                                    const PrivateKey<DCRTPoly>& sk) {
    Plaintext pt;
    cc->Decrypt(ct, sk, &pt);
    return pt->GetRealPackedValue();
}

// decrypt_pt: like decrypt() but returns the Plaintext object itself.
inline Plaintext decrypt_pt(const CC& cc, Ctx ct,
                             const PrivateKey<DCRTPoly>& sk) {
    Plaintext pt;
    cc->Decrypt(ct, sk, &pt);
    return pt;
}

// ── level helpers ─────────────────────────────────────────────────────────

inline uint32_t level_of(const Ctx& ct) {
    return (uint32_t)ct->GetLevel();
}

// Drop ct to the same level as ref.
// fideslib has no LevelReduceInPlace; use the static SetLevel instead.
inline void match_level(const CC& /*cc*/, Ctx& ct, const Ctx& ref) {
    uint32_t ct_lvl  = level_of(ct);
    uint32_t ref_lvl = level_of(ref);
    if (ct_lvl < ref_lvl)
        CryptoContextImpl<DCRTPoly>::SetLevel(ct, ref_lvl);
}

inline void drop_levels(const CC& /*cc*/, Ctx& ct, uint32_t n) {
    if (n > 0)
        CryptoContextImpl<DCRTPoly>::SetLevel(ct, level_of(ct) + n);
}

// Properly reduce ct.consumed to target_level by repeated ct-pt multiply
// (multiply by encoded 1.0) followed by Rescale.
//
// Unlike match_level / drop_levels (metadata-only SetLevel), this actually
// drops polynomial limbs via multPt+rescale so that FIDESlib's GPU
// back-conversion formula (SetLevel = old_level + numElems - numRes)
// stays consistent.  Required when aligning ciphertexts that span a
// bootstrap boundary (large consumed-level gaps).
//
// Note: EvalMultInPlace(ct, double_scalar) calls multScalar() which does NOT
// drop limbs.  Only EvalMult(ct, Plaintext) (multPt) actually does.
inline void reduce_to_level(const CC& cc, Ctx& ct,
                             uint32_t target_level, int slots) {
    while (level_of(ct) < target_level) {
        std::vector<double> ones(slots, 1.0);
        Ptx pt_one = cc->MakeCKKSPackedPlaintext(ones, /*noiseScaleDeg=*/1,
                                                  (uint32_t)level_of(ct));
        cc->EvalMultInPlace(ct, pt_one);
        cc->RescaleInPlace(ct);
    }
}
