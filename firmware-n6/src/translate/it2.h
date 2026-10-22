/**
 * @file     it2.h
 * @brief    English-to-Hindi translation on the STM32N6: an IndicTrans2 (MIT licence) engine.
 * @details  Runs a packed IndicTrans2 en->indic model (it2_pack.py): a pre-norm transformer with
 *           18 encoder layers, a subset of the decoder layers, 4-bit weights in groups of 64 and
 *           Hindi-only output tokens. The model blob is read in place (memory-mapped NOR flash);
 *           all state lives in a caller-provided workspace (it2_workspace_size()).
 *
 *           Pipeline: English text -> Moses-style tokenisation -> SentencePiece BPE -> encoder ->
 *           greedy decoding with a key/value cache -> Hindi tokens -> UTF-8 text.
 *
 *           Portable C99; on Cortex-M55 the matrix kernels use Helium (MVE) when available.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRANSLATE_IT2_H
#define TRANSLATE_IT2_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Longest source, in BPE tokens including the two language tags and </s>. */
#define IT2_MAX_SRC 96U

/** @brief Longest translation, in target tokens. */
#define IT2_MAX_TGT 128U

/** @brief it2_translate() result when the cancel check stopped it. */
#define IT2_CANCELLED (-100)

/** @brief Blob header, as written by it2_pack.py. */
typedef struct
{
    char magic[4]; /* "IT2B" */
    uint32_t version, d_model, n_heads, d_ffn, n_enc, n_dec, group;
    uint32_t src_vocab, tgt_vocab, n_tensors;
    uint32_t tag_src, tag_tgt, bos, pad, eos, unk;
    uint32_t n_src_pieces, src_off, src_str_off, tgt_off, tgt_str_off;
    uint32_t total_size;
} it2_header_t;

/** @brief One entry of the tensor table. kind 0: float32; kind 1: Q4 (see it2_pack.py). */
typedef struct
{
    uint32_t kind, rows, cols, data_off, scale_off;
} it2_tensor_t;

/** @brief Weights of one linear layer: y = W x + b. */
typedef struct
{
    const it2_tensor_t *w;
    const float *b;
} it2_linear_t;

typedef struct
{
    const float *w, *b;
} it2_norm_t;

typedef struct
{
    it2_norm_t sa_norm, ff_norm;
    it2_linear_t q, k, v, o, fc1, fc2;
} it2_enc_layer_t;

typedef struct
{
    it2_norm_t sa_norm, ca_norm, ff_norm;
    it2_linear_t q, k, v, o, cq, ck, cv, co, fc1, fc2;
} it2_dec_layer_t;

/** @brief Polled once per output token; non-zero stops the translation. */
typedef int (*it2_cancel_t)(void);

/** @brief A model opened in place over its blob. */
typedef struct
{
    it2_cancel_t cancel; /* NULL: never; set with it2_set_cancel() after it2_open() */
    const uint8_t *blob;
    const it2_header_t *hdr;
    const it2_tensor_t *src_emb, *tgt_emb;
    it2_norm_t enc_emb_norm, enc_final_norm, dec_emb_norm, dec_final_norm;
    it2_enc_layer_t *enc; /* n_enc entries, in the workspace */
    it2_dec_layer_t *dec; /* n_dec entries, in the workspace */
} it2_model_t;

/** @brief Timing and size of the last translation. */
typedef struct
{
    uint32_t src_tokens, tgt_tokens;
    uint32_t encode_ms, decode_ms;
} it2_stats_t;

/** @brief Millisecond clock used for it2_stats_t; the firmware or host test supplies it. */
uint32_t it2_now_ms(void);

/** @brief Workspace bytes needed for a blob (layer tables, activations, key/value caches). */
size_t it2_workspace_size(const void *blob);

/**
 * @brief Open a model in place.
 * @param[out] m Model.
 * @param[in] blob The packed model; must stay mapped while the model is used.
 * @param[in] workspace it2_workspace_size(blob) bytes, 32-byte aligned, kept for the model's life.
 * @return 0 on success, negative if the blob is not a valid IT2B model.
 */
int it2_open(it2_model_t *m, const void *blob, void *workspace);

/** @brief Let it2_translate() stop early when @p cancel returns non-zero (NULL: never). */
void it2_set_cancel(it2_model_t *m, it2_cancel_t cancel);

/**
 * @brief Translate one English sentence to Hindi.
 * @param[in] m Opened model.
 * @param[in] english UTF-8 English text (one sentence; ASR output with punctuation is fine).
 * @param[out] hindi UTF-8 Hindi text, always NUL-terminated.
 * @param[in] capacity Bytes available at @p hindi.
 * @param[out] stats Optional timing; may be NULL.
 * @return Bytes written (without the NUL), negative on error, IT2_CANCELLED if cancelled.
 */
int it2_translate(const it2_model_t *m, const char *english, char *hindi, size_t capacity, it2_stats_t *stats);

/**
 * @brief Tokenise English into source ids (exposed for testing).
 * @return Number of ids written to @p ids (at most @p max), including the tags and </s>.
 */
uint32_t it2_encode_text(const it2_model_t *m, const char *english, uint32_t *ids, uint32_t max);

#ifdef __cplusplus
}
#endif

#endif /* TRANSLATE_IT2_H */
