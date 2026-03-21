#pragma once
// fideslib_wrapper.h
// Thin convenience layer on top of FIDESlib (OpenFHE 1.4.2 + GPU acceleration).
// Every other file in this project #includes only this header for FHE types.
//
// FIDESlib API:
//   • Standard OpenFHE API for all operations (EvalAdd, EvalMult, EvalRotate …)
//   • cc->LoadContext()  to pin the context to GPU before any computation
//   • EvalBootstrap via the full CKKS bootstrapping circuit (GPU-accelerated)
//
// OpenFHE level convention (FLEXIBLEAUTO):
//   • Fresh ciphertext: GetLevel() == 0  (all moduli available)
//   • After k rescalings: GetLevel() == k
//   • LevelReduce(ct, nullptr, n) increases GetLevel() by n (consumes moduli)
//   • EvalBootstrap() resets level back to ~0

#include <fideslib.hpp>   // brings in all of OpenFHE and FIDESlib headers
#include <openfhe.h>

#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <stdexcept>
#include <memory>

// ── type aliases ──────────────────────────────────────────────────────────
using namespace lbcrypto;

using CC   = CryptoContext<DCRTPoly>;
using Ctx  = Ciphertext<DCRTPoly>;    // shared_ptr<CiphertextImpl<DCRTPoly>>
using Ptx  = Plaintext;               // shared_ptr<PlaintextImpl>
using KP   = KeyPair<DCRTPoly>;

// ── CKKS context bundle ───────────────────────────────────────────────────
// Holds everything needed to run CKKS+bootstrapping on GPU via FIDESlib.
struct CKKSContext {
    CC  cc;
    KP  keys;

    // Convenience aliases
    PublicKey<DCRTPoly>&  pk()  { return keys.publicKey; }
    PrivateKey<DCRTPoly>& sk()  { return keys.secretKey; }
};

// ── parameter factory ────────────────────────────────────────────────────
// Matches the Go code's ParametersLiteral:
//   LogQ=[52,40×16], LogP=[61×3], scale=2^40, LogN=12..16
// For bootstrapping we use FLEXIBLEAUTOEXT (most stable).
inline std::shared_ptr<CKKSContext>
make_ckks_context(int logN                  = 12,
                  int depth                 = 16,   // multiplicative depth
                  int scale_bits            = 40,
                  uint32_t bootstrap_slots  = 0,    // 0 = N/2 (full-packing)
                  bool enable_bootstrap     = true) {
    CCParams<CryptoContextCKKSRNS> params;

    // Bootstrapping depth budget (CoeffToSlot=4, EvalMod=4, SlotToCoeff=4)
    // Must be set before GenCryptoContext when bootstrapping is used.
    std::vector<uint32_t> level_budget = {4, 4};
    uint32_t approx_bootstrap_depth    = 8;   // approximate extra depth for bootstrapping

    params.SetMultiplicativeDepth(depth + (enable_bootstrap ? approx_bootstrap_depth : 0));
    params.SetScalingModSize(scale_bits);
    params.SetFirstModSize(52);               // matches Go LogQ[0]=52
    params.SetScalingTechnique(FLEXIBLEAUTO); // automatic rescaling
    params.SetBatchSize(1 << (logN - 1));     // N/2 slots
    params.SetSecretKeyDist(UNIFORM_TERNARY);
    params.SetNumLargeDigits(3);              // matches LogP[0..2] = 3 special primes
    params.SetKeySwitchTechnique(HYBRID);

    if (enable_bootstrap) {
        // Level budget for the bootstrapping linear transforms
        params.SetLevelBudget(level_budget);
        // Dimension-1 for CoeffToSlot/SlotToCoeff baby-step giant-step
        params.SetBsgsDim({0, 0});
    }

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

    // Pre-generate rotation keys for the rotation steps used in LLaMA.
    // Covers: powers-of-2, steps 5, 1024/j for j=1..256, plus small indices.
    std::vector<int32_t> rot_steps;
    for (int i = 1; i <= 1024; i *= 2) rot_steps.push_back(i);
    for (int i = 1; i <= 1024; i *= 2) rot_steps.push_back(-i);
    for (int j = 1; j <= 256; j *= 2) {
        rot_steps.push_back(1024 / j);
        rot_steps.push_back(-(1024 / j));
    }
    rot_steps.push_back(5);
    rot_steps.push_back(-5);
    // Deduplicate
    std::sort(rot_steps.begin(), rot_steps.end());
    rot_steps.erase(std::unique(rot_steps.begin(), rot_steps.end()), rot_steps.end());
    cc->EvalRotateKeyGen(kp.secretKey, rot_steps);

    if (enable_bootstrap) {
        uint32_t slots = (bootstrap_slots == 0) ? (1u << (logN - 1)) : bootstrap_slots;
        cc->EvalBootstrapSetup(level_budget, {0, 0}, slots);
        cc->EvalBootstrapKeyGen(kp.secretKey, slots);
    }

    // Move the entire context (parameters, evaluation keys, NTT tables) to GPU
    cc->LoadContext();

    auto ctx = std::make_shared<CKKSContext>();
    ctx->cc   = cc;
    ctx->keys = kp;
    return ctx;
}

// ── encode / encrypt helpers ─────────────────────────────────────────────

inline Ptx encode(const CC& cc, const std::vector<double>& values,
                   int level = 0) {
    return cc->MakeCKKSPackedPlaintext(values, /*noiseScaleDeg=*/1,
                                       (uint32_t)level);
}

inline Ptx encode(const CC& cc, const std::vector<std::complex<double>>& values,
                   int level = 0) {
    return cc->MakeCKKSPackedPlaintext(values, /*noiseScaleDeg=*/1,
                                       (uint32_t)level);
}

inline Ptx encode_const(const CC& cc, double val, size_t slots, int level = 0) {
    std::vector<double> v(slots, val);
    return encode(cc, v, level);
}

inline Ctx encrypt(const CC& cc, const Ptx& pt, const PublicKey<DCRTPoly>& pk) {
    return cc->Encrypt(pk, pt);
}

inline Ctx encrypt_const(const CC& cc, double val, size_t slots,
                          const PublicKey<DCRTPoly>& pk, int level = 0) {
    return encrypt(cc, encode_const(cc, val, slots, level), pk);
}

inline std::vector<double> decrypt(const CC& cc, const Ctx& ct,
                                    const PrivateKey<DCRTPoly>& sk) {
    Plaintext pt;
    cc->Decrypt(sk, ct, pt);
    return pt->GetRealPackedValue();
}

// ── level helpers ─────────────────────────────────────────────────────────

// Current "depth consumed" (0 = fresh, increases after each rescale)
inline uint32_t level_of(const Ctx& ct) {
    return ct->GetLevel();
}

// Drop ct to the same level as ref (consume moduli until levels match)
inline void match_level(const CC& cc, Ctx& ct, const Ctx& ref) {
    uint32_t ct_lvl  = level_of(ct);
    uint32_t ref_lvl = level_of(ref);
    if (ct_lvl < ref_lvl)
        cc->LevelReduceInPlace(ct, nullptr, ref_lvl - ct_lvl);
}

// Explicit level drop by n steps
inline void drop_levels(const CC& cc, Ctx& ct, uint32_t n) {
    if (n > 0) cc->LevelReduceInPlace(ct, nullptr, n);
}
