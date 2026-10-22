/**
 * @file     tts_front.h
 * @brief    Hindi text-to-speech front end on the CPU: MMS-TTS (VITS) text encoder and stochastic
 *           duration predictor, giving the prior latents that the flow and the HiFi-GAN vocoder
 *           (on the NPU) turn into audio.
 * @details  Text -> character ids (the MMS tokenizer, blank between characters) -> text encoder
 *           (6 post-norm transformer layers, windowed relative attention) -> prior means and
 *           log-variances per id; the stochastic duration predictor in reverse (rational-quadratic
 *           spline flows) -> frames per id, ceil(exp(logw)); then for every frame
 *           z = m + eps * exp(logs) * noise_scale, 62.5 frames per second of audio.
 *
 *           Weights: tts_front_pack.py (int8 text encoder, float16 duration predictor), read in
 *           place from flash. All state lives in a caller-provided workspace. The noise is a
 *           counter-based generator: frame t, channel c always gets the same value for a seed, so
 *           the latents can be produced in any order or in overlapping windows.
 *
 *           Portable C99; the int8 dot products use Helium (MVE) on the Cortex-M55.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TTS_TTS_FRONT_H
#define TTS_TTS_FRONT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Longest input in ids: 2 x characters + 1 (255 characters). */
#define TTS_MAX_IDS 511U

/** @brief Latent channels per frame. */
#define TTS_CHANNELS 192U

/** @brief Audio samples per frame (16 kHz output, 62.5 frames per second). */
#define TTS_SAMPLES_PER_FRAME 256U

/** @brief Blob header, as written by tts_front_pack.py. */
typedef struct
{
    char magic[4]; /* "TTSF" */
    uint32_t version, hidden, heads, ffn, layers, window;
    uint32_t n_vocab, vocab_off, n_tensors, table_off, total_size;
    float noise_scale, noise_scale_duration, length_scale;
} tts_header_t;

/** @brief One entry of the tensor table. kind 0: float32, 1: float16, 3: int8 in groups of 64. */
typedef struct
{
    uint32_t kind, rows, cols, data_off, scale_off;
} tts_tensor_t;

/** @brief A front end opened over its blob, with the result of the last tts_front_run(). */
typedef struct
{
    const uint8_t *blob;
    const tts_header_t *hdr;
    const tts_tensor_t *t;
    void *ws;          /* workspace */
    uint32_t n_ids;    /* ids of the last run */
    uint32_t n_frames; /* frames of the last run */
    uint32_t seed;     /* noise seed of the last run */
} tts_front_t;

/** @brief Workspace bytes tts_front_open() needs (activations for TTS_MAX_IDS ids). */
size_t tts_front_workspace_size(void);

/**
 * @brief Open a front end in place.
 * @param[in] blob tts_front_pack.py output; must stay mapped while used.
 * @param[in] workspace tts_front_workspace_size() bytes, 32-byte aligned, kept while used.
 * @return 0 on success, negative if the blob is not a TTSF front end this code reads.
 */
int tts_front_open(tts_front_t *f, const void *blob, void *workspace);

/**
 * @brief Hindi UTF-8 text to ids, as the MMS tokenizer does: characters outside the model's
 *        alphabet (punctuation, Latin letters, most digits) are dropped, a blank (id 0) goes
 *        between characters and at both ends. Precomposed nukta letters are first decomposed
 *        (U+0958..U+095F, U+0929, U+0931, U+0934), and the vowel letters the model lacks are
 *        spelled as it can say them (ऋ -> रि, ॠ -> री).
 * @return Ids written (at most @p max); 0 if there is nothing to say.
 */
uint32_t tts_front_text_to_ids(const tts_front_t *f, const char *utf8, uint16_t *ids, uint32_t max);

/**
 * @brief Run the text encoder and the duration predictor on ids.
 * @param[in] seed Noise seed (any value; the same seed gives the same voice output).
 * @return Frames of audio (TTS_SAMPLES_PER_FRAME samples each), 0 on error.
 */
uint32_t tts_front_run(tts_front_t *f, const uint16_t *ids, uint32_t n, uint32_t seed);

/** @brief Frames given to each id by the last run (n_ids values). */
const uint16_t *tts_front_durations(const tts_front_t *f);

/**
 * @brief Prior latents of frames [start, start + count) of the last run.
 * @param[out] z z[c * c_stride + i * t_stride] = channel c of frame start + i. Frames at or after
 *             n_frames (padding of a fixed-size window) are written as 0.
 */
void tts_front_latents(const tts_front_t *f, uint32_t start, uint32_t count, float *z, uint32_t c_stride,
                       uint32_t t_stride);

/** @brief Debug and test access: text-encoder outputs of the last run, [n_ids][192] each. */
const float *tts_front_prior_mean(const tts_front_t *f);
const float *tts_front_prior_logvar(const tts_front_t *f);
const float *tts_front_logw(const tts_front_t *f);

/**
 * @brief The noise generator: a standard normal value for (seed, stream, index). Stream 0 is
 *        the duration predictor (index = channel * TTS_MAX_IDS + id), stream 1 the prior
 *        (index = frame * TTS_CHANNELS + channel). Exposed so tests can reproduce it.
 */
float tts_front_gauss(uint32_t seed, uint32_t stream, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif /* TTS_TTS_FRONT_H */
