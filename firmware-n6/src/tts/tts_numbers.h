/**
 * @file     tts_numbers.h
 * @brief    Spells the numbers in Hindi text as words, for the voice.
 * @details  The MMS-TTS Hindi alphabet has only the digits 0-4 and 8, and the translator sometimes
 *           writes numbers as digits ("वह 25 साल की है"). Before speaking, every run of digits
 *           (ASCII or Devanagari) becomes Hindi words in the Indian system: 25 -> पच्चीस,
 *           2,500 -> दो हज़ार पाँच सौ, 1,50,000 -> एक लाख पचास हज़ार; 3.5 -> तीन दशमलव पाँच;
 *           25% -> पच्चीस प्रतिशत. Runs of more than 9 digits (phone numbers) are read digit by
 *           digit. Everything else is copied unchanged.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TTS_TTS_NUMBERS_H
#define TTS_TTS_NUMBERS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copy UTF-8 @p in to @p out with the numbers spelled out.
 * @return Bytes written (without the NUL); the output is cut at @p cap - 1 and always terminated.
 */
uint32_t tts_spell_numbers(const char *in, char *out, uint32_t cap);

#ifdef __cplusplus
}
#endif

#endif /* TTS_TTS_NUMBERS_H */
