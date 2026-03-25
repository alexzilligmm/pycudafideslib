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

#include <fideslib.hpp>

#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <stdexcept>
#include <memory>

using namespace fideslib;

using CC   = CryptoContext<DCRTPoly>;            // shared_ptr<CryptoContextImpl<DCRTPoly>>
using Ctx  = Ciphertext<DCRTPoly>;               // shared_ptr<CiphertextImpl<DCRTPoly>>
using Ptx  = Plaintext;                          // shared_ptr<PlaintextImpl>
using KP   = KeyPair<DCRTPoly>;

struct CKKSContext {
    CC  cc;
    KP  keys;

    PublicKey<DCRTPoly>&  pk()  { return keys.publicKey; }
    PrivateKey<DCRTPoly>& sk()  { return keys.secretKey; }
};

inline std::shared_ptr<CKKSContext>
make_ckks_context(int logN                          = 12,
                  int depth                         = 16,
                  int scale_bits                    = 40,
                  uint32_t bootstrap_slots          = 0,    // 0 = N/2
                  bool enable_bootstrap             = true,
                  int btp_scale_bits                = 59,   // per-level modulus in bootstrap path
                  int first_mod_bits                = 60,   // special first prime size
                  std::vector<uint32_t> level_budget_in = {},  // {} → {3,3} when bootstrapping
                  uint32_t batch_size               = 0,    // 0 = N/2
                  // --- alignment params (match Lattigo ring.Ternary{H} / LogP) ---
                  int h_weight                      = 0,    // 0 = UNIFORM_TERNARY; >0 = SPARSE_TERNARY
                                                            // NOTE: OpenFHE does not enforce exact H —
                                                            // SPARSE_TERNARY is the closest approximation
                  uint32_t num_large_digits         = 3,    // #P primes for hybrid key-switch (Go LogP.size())
                  uint32_t btp_depth_overhead       = 9,    // extra levels reserved for EvalBootstrap
                  std::vector<int32_t> extra_rot_steps = {}) // additional rotation keys (e.g. for linear layer)
{
    CCParams<CryptoContextCKKSRNS> params;

    std::vector<uint32_t> level_budget;
    int actual_scale_bits;
    if (enable_bootstrap) {
        actual_scale_bits = btp_scale_bits;
        level_budget      = level_budget_in.empty() ? std::vector<uint32_t>{3, 3}
                                                    : level_budget_in;
    } else {
        actual_scale_bits    = scale_bits;
        level_budget         = {};
        btp_depth_overhead   = 0;
    }

    const uint32_t slots = (batch_size == 0) ? (1u << (logN - 1)) : batch_size;

    params.SetMultiplicativeDepth(depth + btp_depth_overhead);
    params.SetScalingModSize(actual_scale_bits);
    params.SetFirstModSize(first_mod_bits);
    params.SetScalingTechnique(FLEXIBLEAUTO);
    params.SetBatchSize(slots);
    params.SetSecretKeyDist(h_weight > 0 ? SPARSE_TERNARY : UNIFORM_TERNARY);
    params.SetNumLargeDigits(num_large_digits);
    params.SetKeySwitchTechnique(HYBRID);
    if (enable_bootstrap) {
        params.SetSecurityLevel(HEStd_NotSet);
        params.SetRingDim(1 << logN);
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

    std::vector<int32_t> rot_steps;
    for (int i = 1; i <= (int)slots; i *= 2) rot_steps.push_back(i);
    for (int i = 1; i <= (int)slots; i *= 2) rot_steps.push_back(-i);
    for (int j = 1; j <= 256; j *= 2) {
        rot_steps.push_back(1024 / j);
        rot_steps.push_back(-(1024 / j));
    }
    rot_steps.push_back(5);
    rot_steps.push_back(-5);
    for (auto r : extra_rot_steps) rot_steps.push_back(r);
    std::sort(rot_steps.begin(), rot_steps.end());
    rot_steps.erase(std::unique(rot_steps.begin(), rot_steps.end()), rot_steps.end());
    cc->EvalRotateKeyGen(kp.secretKey, rot_steps);

    if (enable_bootstrap) {
        uint32_t btp_slots = (bootstrap_slots == 0) ? slots : bootstrap_slots;
        // fideslib EvalBootstrapSetup: (levelBudget, dim1, slots, correctionFactor)
        cc->EvalBootstrapSetup(level_budget, {0, 0}, btp_slots, /*correctionFactor=*/0);
        cc->EvalBootstrapKeyGen(kp.secretKey, btp_slots);
    }

    cc->LoadContext(kp.publicKey);

    auto ctx  = std::make_shared<CKKSContext>();
    ctx->cc   = cc;
    ctx->keys = std::move(kp);
    return ctx;
}

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

inline Ctx encrypt(const CC& cc, Ptx pt, const PublicKey<DCRTPoly>& pk) {
    return cc->Encrypt(pk, pt);
}

inline Ctx encrypt_const(const CC& cc, double val, size_t slots,
                          const PublicKey<DCRTPoly>& pk, int level = 0) {
    return encrypt(cc, encode_const(cc, val, slots, level), pk);
}

inline std::vector<double> decrypt(const CC& cc, Ctx ct,
                                    const PrivateKey<DCRTPoly>& sk) {
    Plaintext pt;
    cc->Decrypt(ct, sk, &pt);
    return pt->GetRealPackedValue();
}

inline Plaintext decrypt_pt(const CC& cc, Ctx ct,
                             const PrivateKey<DCRTPoly>& sk) {
    Plaintext pt;
    cc->Decrypt(ct, sk, &pt);
    return pt;
}

inline uint32_t level_of(const Ctx& ct) {
    return (uint32_t)ct->GetLevel();
}

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
