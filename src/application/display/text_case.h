/**
 * @file     text_case.h
 * @brief    Sentence case for the recognized text, which the network spells in capitals.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_DISPLAY_TEXT_CASE_H
#define APPLICATION_DISPLAY_TEXT_CASE_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief Rewrite text in sentence case, in place: letters in lower case, except the first letter of
 *        a sentence and the word I (also in I'M, I'LL, I'D, I'VE).
 * @param[in,out] text Characters to rewrite; letters may be in either case, other bytes are kept.
 * @param[in] length Characters in @p text; no terminator is needed.
 * @param[in] sentence_start Non-zero when @p text begins a sentence.
 * @note A sentence starts after a full stop or a question mark. The last character is taken to end
 *       its word, so a trailing "I" is a capital until the next letters arrive; the caller re-runs
 *       this on every redraw over the whole text, so it settles once they do.
 */
void text_sentence_case(char *text, uint32_t length, int sentence_start);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_DISPLAY_TEXT_CASE_H */
