/**
 * @file     xl_speech.h
 * @brief    Speaks Hindi text on the translator's speaker: MMS-TTS front end on the CPU
 *           (tts_front.c), flow and vocoder on the NPU (xl_voice_net.c), 16 -> 48 kHz
 *           (xl_resample.c), into the speaker ring (xl_audio_out.c).
 * @details  The flow runs over the whole sentence in 512-frame windows; then the vocoder makes
 *           1.024 s of audio per run. Audio is queued as it is made, once two runs are ahead (or
 *           the sentence is done), so speech starts after about one second. The caller's cancel
 *           check is polled between steps and while waiting for room in the ring: when music
 *           takes the speaker, speech stops within one step (the display thread flushes the ring
 *           at once).
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_SPEECH_H
#define APPLICATION_TRANSLATOR_XL_SPEECH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Longest speech, in latent frames (62.5 per second): 32 s. */
#define XL_SPEECH_MAX_FRAMES 2048U

/** @brief Polled while speaking; non-zero stops the speech. */
typedef int (*xl_speech_cancel_t)(void);

/** @brief Workspace xl_speech_init() needs. */
size_t xl_speech_workspace_size(void);

/**
 * @brief Open the voice: front-end weights at @p front_blob (flash), the NPU networks.
 * @param[in] workspace xl_speech_workspace_size() bytes, 32-byte aligned (PSRAM).
 * @return 0 if speech works, negative if not (missing networks or weights: text only).
 * @note Call from the translator thread after the recognizer started (it owns the NPU runtime).
 */
int xl_speech_init(const void *front_blob, void *workspace);

/** @brief Non-zero after a successful xl_speech_init(). */
int xl_speech_ready(void);

/**
 * @brief Speak @p hindi; returns when the audio has been played (or cancelled).
 * @param[in] seed Noise seed (any value).
 * @param[in] cancel Polled between steps; may be NULL.
 * @return 0 when spoken, 1 when cancelled, negative on error.
 * @pre The recognizer is paused (audio_port_set_paused) and the NPU taken
 *      (asr_engine_npu_acquire); the speaker stream runs.
 */
int xl_speech_say(const char *hindi, uint32_t seed, xl_speech_cancel_t cancel);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_SPEECH_H */
