/**
 * @file     xl_audio_out.h
 * @brief    Translator speaker output: SAI2 block B as I2S master into the MAX98357A, through
 *           the TMUX1574 audio switch (its B side).
 * @details  48 kHz, 16-bit, both I2S slots carrying the same mono sample (the MAX98357A plays the
 *           (L+R)/2 mix). Pins on the Arduino header: PE12 SAI2_SCK_B (A2), PE13 SAI2_FS_B (D6),
 *           PE7 SAI2_SD_B (D8). The application writes mono 48 kHz samples into a ring
 *           (xl_audio_out_write()); the DMA interrupt moves them to the I2S stream and plays
 *           silence when the ring is empty.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_AUDIO_OUT_H
#define APPLICATION_TRANSLATOR_XL_AUDIO_OUT_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                                PUBLIC DEFINES                               */
/*******************************************************************************/

/** @brief Sample rate of the samples xl_audio_out_write() takes (the I2S rate, nominal). */
#define XL_AUDIO_RATE_HZ 48000U

/**
 * @brief Level ceiling applied to every written sample, Q15 (23198 = -3 dB).
 * @details Sized for an 8 Ohm 0.5 W speaker on a MAX98357A at 5 V with GAIN at 3 dB (100 kOhm
 *          from GAIN to VIN). The amplifier plays a full-scale sine at 2.1 dBV + 3 dB = 1.8 Vrms,
 *          0.40 W into 8 Ohm. After this ceiling a full-scale sine gives 0.20 W, and even a
 *          full-scale square wave stays at 0.40 W. With GAIN left open (9 dB) the same sine would
 *          be 1.6 W, so the GAIN resistor is required. Override with -DXL_AUDIO_OUT_CEILING_Q15=n.
 */
#ifndef XL_AUDIO_OUT_CEILING_Q15
#define XL_AUDIO_OUT_CEILING_Q15 23198
#endif

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief Clock, pins, SAI2 and GPDMA1 channel 7; nothing is transmitted yet.
 * @return 0 on success, a negative step number on failure.
 * @note The SAI kernel clock is IC8 from PLL3, the microphone's audio clock: started here if the
 *       microphone has not started it yet, with the microphone's settings.
 */
int xl_audio_out_init(void);

/**
 * @brief Start the I2S stream (silence until samples are written). No effect if running.
 * @return 0 on success, negative on failure.
 */
int xl_audio_out_start(void);

/** @brief Stop the I2S stream; the clocks stop, the queued samples are kept. */
void xl_audio_out_stop(void);

/** @brief Non-zero while the I2S stream runs. */
int xl_audio_out_running(void);

/**
 * @brief Queue mono 48 kHz samples for playback, scaled by XL_AUDIO_OUT_CEILING_Q15.
 * @param[in] samples PCM16 samples.
 * @param[in] count Samples in @p samples.
 * @return Samples accepted; fewer than @p count when the ring (4 s) is full.
 */
uint32_t xl_audio_out_write(const int16_t *samples, uint32_t count);

/** @brief Samples queued and not played yet. */
uint32_t xl_audio_out_queued(void);

/** @brief Drop every queued sample (playback goes silent within one DMA half, about 5 ms). */
void xl_audio_out_flush(void);

/**
 * @brief Queue a sine tone with 5 ms fades, for checking the wiring.
 * @param[in] frequency_hz Tone frequency.
 * @param[in] duration_ms Tone length.
 * @param[in] amplitude Peak amplitude, 0 to 32767.
 * @return Samples queued.
 */
uint32_t xl_audio_out_tone(uint32_t frequency_hz, uint32_t duration_ms, int16_t amplitude);

/** @brief DMA halves played with an empty ring since start-up (silence was sent). */
uint32_t xl_audio_out_underruns(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_AUDIO_OUT_H */
