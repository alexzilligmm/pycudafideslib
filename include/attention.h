#pragma once

#include "inference.h"
#include "nonlinear.h"
#include <vector>

// Rearrange weight columns from head-grouped [head0_dims | head1_dims | ...]
// to interleaved [h0_d0, h1_d0, ..., hH_d0, h0_d1, h1_d1, ...]
// Matches Python rearrange_q_k_v(W, H)
std::vector<std::vector<double>> rearrange_qkv_weights(
    const std::vector<std::vector<double>>& W, int H);

// Precompute masks needed for MHA operations (tok0 mask, init K cache)
void prepare_mha_masks(Inference& inf);

// Push one key ciphertext into K cache
void cache_k_push(Inference& inf, const Ctx& key_ct);

// Compute QK^T against cached keys, returns scaled scores
Ctx qkt(Inference& inf, const Ctx& query_ct);

// Intra-head sum reduction + inter-block aggregation
Ctx head_reduce_sum(Inference& inf, const Ctx& ct);

// Attention-specific softmax with oracle max and head-interleaved layout
Ctx attention_softmax(Inference& inf, const Ctx& scores,
                      int num_keys, double given_max,
                      const SoftmaxConfig& cfg = SOFTMAX_ENCLLM_GPT2);

// Initialize V cache metaciphertexts (d_head zero ciphertexts)
void prepare_vcache(Inference& inf);

// Push one value ciphertext into V cache
void cache_v_push(Inference& inf, const Ctx& value_ct);

// Compute weighted V sum given softmax scores
Ctx softmax_v(Inference& inf, const Ctx& softmax_scores);
