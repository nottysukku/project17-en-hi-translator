/**
 * @file     xl_sentence.h
 * @brief    Cuts the recognized English into sentences for the translator.
 * @details  The recognizer publishes text every half second: newly committed characters, and a
 *           tentative tail for the audio it is still unsure about. Committed text collects here
 *           until a sentence is complete:
 *           - at a committed end mark ('.', '?' or '!'); the punctuation model rarely writes
 *             them, so also
 *           - after a pause: nothing new committed for XL_SENTENCE_PAUSE_MS and no tentative
 *             text (the speaker stopped talking), or
 *           - when it grows past XL_SENTENCE_MAX_CHARS: cut at the last comma or space.
 *           Plain C, no RTOS: the caller passes the time. Not thread-safe; the display thread
 *           feeds it and takes the sentences.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_SENTENCE_H
#define APPLICATION_TRANSLATOR_XL_SENTENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Silence that ends a sentence, in milliseconds. */
#ifndef XL_SENTENCE_PAUSE_MS
#define XL_SENTENCE_PAUSE_MS 1200U
#endif

/** @brief A sentence longer than this is cut (the translator takes about 90 BPE tokens). */
#define XL_SENTENCE_MAX_CHARS 200U

/** @brief Room for one sentence, with its terminator. */
#define XL_SENTENCE_CAPACITY (XL_SENTENCE_MAX_CHARS + 2U)

/** @brief Forget any pending text (for example when music takes the speaker). */
void xl_sentence_reset(void);

/**
 * @brief Take newly recognized text.
 * @param[in] committed Newly committed characters (need not be terminated).
 * @param[in] length Characters in @p committed.
 * @param[in] tentative Current tentative tail, NUL-terminated (may be NULL).
 * @param[in] now_ms Milliseconds since start-up.
 */
void xl_sentence_feed(const char *committed, uint32_t length, const char *tentative, uint32_t now_ms);

/**
 * @brief The next complete sentence, if any.
 * @param[out] out The sentence, trimmed and NUL-terminated, as recognized (upper case).
 * @param[in] capacity Bytes at @p out (XL_SENTENCE_CAPACITY is enough).
 * @param[in] now_ms Milliseconds since start-up.
 * @return Characters written, 0 if no sentence is complete yet.
 */
uint32_t xl_sentence_take(char *out, uint32_t capacity, uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_SENTENCE_H */
