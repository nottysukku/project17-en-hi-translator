/**
 * @file     xl_audio_out.h
 * @brief    Translator speech audio output interface.
 * @details  Hindi translation speech output:
 *           - Target output: Onboard 3.5 mm audio jack into an external PAM8403 Class-D
 *             amplifier module (with a 470 uF decoupling capacitor across +5V and GND near PAM)
 *             driving a 4 Ohm 3 W speaker.
 *           - Digital I2S option: SAI2 block B as I2S master into MAX98357A through the
 *             TMUX1574 audio switch (side B).
 *           48 kHz, 16-bit, both slots carrying the same mono sample. Pins on Arduino header
 *           for SAI2: PE12 SAI2_SCK_B (A2), PE13 SAI2_FS_B (D6), PE7 SAI2_SD_B (D8).
 *           The application writes mono 48 kHz samples into a ring (xl_audio_out_write());
 *           the DMA interrupt moves them to the audio stream and plays silence when empty.
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
 * @details Sized for safe output headroom without clipping on the PAM8403 / 4 Ohm 3 W speaker,
 *          or an 8 Ohm 0.5 W speaker on a MAX98357A at 5 V with GAIN at 3 dB (100 kOhm
 *          from GAIN to VIN). Override with -DXL_AUDIO_OUT_CEILING_Q15=n.
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
 * @param samples Mono int16 samples.
 * @param count Number of samples.
 * @return Samples actually accepted; less than @p count if the ring became full.
 */
uint32_t xl_audio_out_write(const int16_t *samples, uint32_t count);

/** @brief Number of mono samples currently queued in the ring. */
uint32_t xl_audio_out_queued(void);

/** @brief Discard all queued samples; the stream continues from silence. */
void xl_audio_out_flush(void);

/**
 * @brief Synthesize a sine wave of the given frequency and length into the ring.
 * @param frequency_hz Frequency in Hz (e.g. 880).
 * @param duration_ms Duration in milliseconds.
 * @param amplitude Peak amplitude (before the ceiling), 0..32767.
 * @return Samples written.
 */
uint32_t xl_audio_out_tone(uint32_t frequency_hz, uint32_t duration_ms, int16_t amplitude);

/** @brief DMA transfers that found the ring empty while speech was in flight. */
uint32_t xl_audio_out_underruns(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_AUDIO_OUT_H */
