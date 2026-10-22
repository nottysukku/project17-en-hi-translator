/**
 * @file     xl_display.h
 * @brief    Translator screen and power states, run by the display thread
 *           (-DAPP_TRANSLATOR_MODE=ON).
 * @details  The screen shows which board drives the speaker, the recognized English, the Hindi
 *           translation and four touch buttons: AUTO, TRANSLATE, MUSIC (the router modes) and TONE
 *           (a test tone on the speaker, for checking the wiring). While the WBA music has the
 *           speaker the N6 goes to sleep: speech recognition paused, speaker output stopped, screen
 *           and backlight off. A touch or the USER button lights the screen for 10 s; music
 *           stopping (AUTO) or a tap on TRANSLATE wakes the N6 fully.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_DISPLAY_H
#define APPLICATION_TRANSLATOR_XL_DISPLAY_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>

#include "tx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief Bring up the router pins, the speaker output and the USER button; the screen follows
 *        after the start-up logo. Call from the display thread.
 */
void xl_display_start(void);

/**
 * @brief Take recognized text; cheap (copies only), so it may run with the mailbox locked.
 * @param[in] final_text Newly committed characters, need not be terminated.
 * @param[in] final_length Characters in @p final_text.
 * @param[in] tentative Newest tentative tail, NUL-terminated.
 */
void xl_display_text(const char *final_text, uint32_t final_length, const char *tentative);

/** @brief Read the inputs, run the router and the power states, redraw if needed. */
void xl_display_poll(void);

/** @brief How long the display thread may wait before the next xl_display_poll(), in ticks. */
ULONG xl_display_poll_ticks(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_DISPLAY_H */
