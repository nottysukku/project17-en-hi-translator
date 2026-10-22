/**
 * @file     xl_translator.h
 * @brief    The translator thread: English sentences in; Hindi text (for the screen) and Hindi
 *           speech (on the speaker) out.
 * @details  Runs below the recognizer and the display (APP_TRANSLATOR_THREAD_PRIORITY). For each
 *           sentence: translate (it2.c, model in NOR flash), publish the Hindi, then, if the N6
 *           has the speaker, speak it (xl_speech.c) with recognition paused and the NPU borrowed
 *           from the recognizer. The translation workspace and the voice share one PSRAM region,
 *           used in turn (xl_models.h).
 *
 *           Music always wins: xl_translator_cancel() (the display thread calls it when the WBA
 *           takes the speaker) stops a translation within one token, speech within one vocoder
 *           step, and drops the queued sentences.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_TRANSLATOR_H
#define APPLICATION_TRANSLATOR_XL_TRANSLATOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Reasons for pausing recognition; it listens only when none applies. */
#define XL_LISTEN_MUSIC  0x1U /**< the WBA has the speaker */
#define XL_LISTEN_SPEECH 0x2U /**< the N6 is speaking (it would hear itself) */

/** @brief Hindi text kept for the screen, bytes. */
#define XL_TRANSLATOR_HINDI_CAPACITY 1024U

/** @brief Create the thread and its RTOS objects (before the scheduler). */
uint32_t xl_translator_create(void);

/** @brief Start the thread (after the recognizer has started the NPU runtime). */
uint32_t xl_translator_start(void);

/**
 * @brief Queue an English sentence (as recognized, any case). Cheap; when the queue is full the
 *        oldest sentence is dropped.
 */
void xl_translator_submit(const char *english);

/** @brief Stop the current translation and speech now, and drop the queued sentences. */
void xl_translator_cancel(void);

/**
 * @brief Pause or resume recognition for one reason (XL_LISTEN_*); safe from any thread.
 * @param[in] pause Non-zero to add the reason, zero to remove it.
 */
void xl_translator_listen(uint32_t reason, int pause);

/**
 * @brief The latest Hindi translation, if it changed.
 * @param[in,out] revision The caller's last seen revision; updated.
 * @return Non-zero when @p out was filled with a newer translation.
 */
int xl_translator_hindi(char *out, uint32_t capacity, uint32_t *revision);

/** @brief What the thread is doing, for the status line: 0 idle, 1 translating, 2 speaking. */
uint32_t xl_translator_activity(void);

/** @brief Non-zero when the translation model is installed (flashed and valid). */
int xl_translator_model_ok(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_TRANSLATOR_H */
