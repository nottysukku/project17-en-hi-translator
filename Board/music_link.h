/**
 * @file     music_link.h
 * @brief    Music level and the "music playing" line to the STM32N6 (Project 17).
 * @details  - Volume: the MAX98357A has no volume control, so the phone's volume (VCP, mapped by
 *             the application to 0..100) is applied here to every decoded sample, under a fixed
 *             -3 dB ceiling sized for the 8 Ohm 0.5 W speaker (GAIN at 3 dB).
 *           - Music line: PB0 (Arduino D6, CN8-7) is driven high while a stream plays audible
 *             sound, and low within 0.1 s after the stream stops or after 1 s of silence. The N6
 *             reads it on PD0 (its D2) and hands over the speaker.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef MUSIC_LINK_H
#define MUSIC_LINK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Highest output level, Q15 (23198 = -3 dB): a full-scale sine is then 0.20 W into 8 Ohm. */
#define MUSIC_LINK_CEILING_Q15   23198

/** @brief Attenuation at volume 1 (of 100), in dB below the ceiling; volume 0 is silence. */
#define MUSIC_LINK_RANGE_DB      50.0f

/** @brief Peak sample magnitude that counts as audible (about -54 dBFS). */
#define MUSIC_LINK_AUDIBLE_PEAK  64

/** @brief Silence (or volume 0) that ends "music playing", in ms. */
#define MUSIC_LINK_SILENT_MS     1000U

/** @brief Poll period of the line, in ms; a stream without frames for this long has stopped. */
#define MUSIC_LINK_POLL_MS       100U

/** @brief PB0 as a push-pull output, low, and the poll timer. Call once at start-up. */
void MUSIC_LINK_Init(void);

/** @brief Set the volume, 0..100 (the DK1 BSP scale; 0 is silence). */
void MUSIC_LINK_SetVolume(uint32_t percent);

/**
 * @brief Scale one channel of a decoded frame in place and track its level.
 * @param[in,out] samples First sample of the channel in the interleaved stereo buffer.
 * @param[in] frames Samples of this channel; they sit at even indices (stride 2).
 */
void MUSIC_LINK_ProcessChannel(int16_t *samples, uint32_t frames);

/** @brief The stream stopped: drop the line now. */
void MUSIC_LINK_StreamStopped(void);

#ifdef __cplusplus
}
#endif

#endif /* MUSIC_LINK_H */
