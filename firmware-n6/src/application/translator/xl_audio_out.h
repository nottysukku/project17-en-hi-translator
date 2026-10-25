/**
 * @file     xl_audio_out.h
 * @brief    Translator audio: the DK's 3.5 mm jack plays the Hindi voice or the WBA's music.
 * @details  Output: SAI1 block A as I2S master with MCLK into the DK's WM8904 codec (I2C2 address
 *           0x34), whose headphone output is the 3.5 mm jack. On the jack: a PAM8403 amplifier
 *           module (470 uF across its 5 V input) and a 4 Ohm 3 W speaker. 48 kHz (47.99 kHz
 *           exactly), 16-bit, the same mono sample on both channels.
 *
 *           Music input: the NUCLEO-WBA55CG sends its decoded music as I2S (it is the clock
 *           master) through the TMUX1574 to SAI2 block B, a slave receiver on the Arduino header:
 *           PE12 SCK (A2), PE13 FS (D6), PE7 SD (D8). The left and right samples are mixed to mono
 *           into a ring and played at the jack's rate: the reading speed follows the ring's level,
 *           so the two boards' clocks may differ (any rate from 8 to 48 kHz is detected).
 *
 *           Which of the two the jack plays is chosen with xl_audio_out_select(): the change fades
 *           out over 5 ms, switches, and fades in, inside the DMA interrupt. The speech ring is
 *           dropped when the music takes over.
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

/** @brief Sample rate of the samples xl_audio_out_write() takes (the jack's rate, nominal). */
#define XL_AUDIO_RATE_HZ 48000U

/**
 * @brief WM8904 headphone volume in dB (-57 to +6), the level at the jack.
 * @details The PAM8403 has a fixed gain of 24 dB (x15.8). It clips near 3.2 Vrms at 5 V into
 *          4 Ohm, which is about 2.5 W. At -14 dB a full-scale sine leaves the jack at about
 *          0.2 Vrms and reaches that edge; speech and music peaks stay below the 3 W rating.
 *          Lower it (-20) if the speaker distorts or the board resets on loud music; the PAM8403
 *          module's own knob, if it has one, works too. Override with -DXL_AUDIO_OUT_VOLUME_DB=n.
 */
#ifndef XL_AUDIO_OUT_VOLUME_DB
#define XL_AUDIO_OUT_VOLUME_DB (-14)
#endif

/**
 * @brief Level ceiling applied to every written speech sample, Q15 (23198 = -3 dB).
 * @details Keeps the Hindi voice 3 dB under the level the volume above allows, so its peaks do
 *          not clip in the PAM8403. The music is scaled by the WBA itself.
 *          Override with -DXL_AUDIO_OUT_CEILING_Q15=n.
 */
#ifndef XL_AUDIO_OUT_CEILING_Q15
#define XL_AUDIO_OUT_CEILING_Q15 23198
#endif

/*******************************************************************************/
/*                                 PUBLIC TYPES                                */
/*******************************************************************************/

/** @brief What the jack plays. */
typedef enum
{
    XL_AUDIO_SPEECH = 0, /**< The translator's ring: Hindi voice, tones. */
    XL_AUDIO_MUSIC       /**< The WBA's music, received on SAI2 block B. */
} xl_audio_source_t;

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief Clocks, pins, both SAI blocks, two GPDMA1 channels (7 out, 5 in) and the WM8904.
 * @return 0 on success; -1 audio clock, -2 DMA, -3 SAI, -4 no WM8904 answered on I2C2 (an older
 *         DK has a CS42L51 instead, not supported), -5 the codec did not take its settings.
 * @note The SAI kernel clock is IC8 from PLL3, the microphone's audio clock: started here if the
 *       microphone has not started it yet, with the microphone's settings. Takes about 0.4 s
 *       (the codec's reference and DC servo settle); call it from the thread that uses I2C2
 *       for the touch screen.
 */
int xl_audio_out_init(void);

/**
 * @brief Start the jack (silence until something is written or selected) and the music input.
 * @return 0 on success, negative on failure. No effect if running.
 */
int xl_audio_out_start(void);

/** @brief Stop both streams; the queued speech is kept. */
void xl_audio_out_stop(void);

/** @brief Non-zero while the jack's stream runs. */
int xl_audio_out_running(void);

/** @brief Choose what the jack plays; the change takes about 10 ms (fade out, fade in). */
void xl_audio_out_select(xl_audio_source_t source);

/** @brief What the jack plays now (the previous choice until the fade-out has ended). */
xl_audio_source_t xl_audio_out_selected(void);

/**
 * @brief Housekeeping of the music input; call every few tens of milliseconds.
 * @details Measures the WBA's sample rate once a second and restarts the receiver after an
 *          error (the WBA stopping in the middle of a frame).
 * @param[in] now_ms Milliseconds since start-up.
 */
void xl_audio_out_poll(uint32_t now_ms);

/** @brief The WBA's sample rate in Hz as last measured; 0 while no music clock arrives. */
uint32_t xl_audio_out_music_rate(void);

/**
 * @brief Queue mono 48 kHz speech samples, scaled by XL_AUDIO_OUT_CEILING_Q15.
 * @param[in] samples PCM16 samples.
 * @param[in] count Samples in @p samples.
 * @return Samples accepted; fewer than @p count when the ring (4 s) is full.
 */
uint32_t xl_audio_out_write(const int16_t *samples, uint32_t count);

/** @brief Speech samples queued and not played yet. */
uint32_t xl_audio_out_queued(void);

/** @brief Drop every queued speech sample (silent within one DMA half, about 5 ms). */
void xl_audio_out_flush(void);

/**
 * @brief Queue a sine tone with 5 ms fades, for checking the wiring.
 * @param[in] frequency_hz Tone frequency.
 * @param[in] duration_ms Tone length.
 * @param[in] amplitude Peak amplitude, 0 to 32767.
 * @return Samples queued.
 */
uint32_t xl_audio_out_tone(uint32_t frequency_hz, uint32_t duration_ms, int16_t amplitude);

/** @brief DMA halves that ran out of speech samples in the middle (silence was sent). */
uint32_t xl_audio_out_underruns(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_AUDIO_OUT_H */
