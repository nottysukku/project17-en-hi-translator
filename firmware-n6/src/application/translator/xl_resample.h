/**
 * @file     xl_resample.h
 * @brief    16 kHz speech to the 48 kHz speaker stream: 3x polyphase interpolation.
 * @details  The voice (MMS-TTS) is 16 kHz; xl_audio_out plays 48 kHz. Each input sample becomes
 *           three output samples through a 96-tap low-pass (passband 0-7 kHz within 0.01 dB,
 *           -64 dB from 9 kHz; model-tools/resample_design.py). Streaming: the state carries the
 *           last 31 input samples between calls.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_RESAMPLE_H
#define APPLICATION_TRANSLATOR_XL_RESAMPLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XL_RESAMPLE_PHASES 3U  /**< Output samples per input sample. */
#define XL_RESAMPLE_TAPS   32U /**< Filter taps per phase. */

/** @brief Interpolator state. */
typedef struct
{
    float history[XL_RESAMPLE_TAPS]; /* newest last */
} xl_resample_t;

/** @brief Start a new stream (silence before it). */
void xl_resample_reset(xl_resample_t *r);

/**
 * @brief Interpolate @p n input samples to 3 n output samples.
 * @param[in] in 16 kHz samples, full scale +-1.0.
 * @param[in] gain Output = gain * 32767 * signal, rounded and clipped to int16.
 * @param[out] out 3 n samples at 48 kHz.
 */
void xl_resample_run(xl_resample_t *r, const float *in, uint32_t n, float gain, int16_t *out);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_RESAMPLE_H */
