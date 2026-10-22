/**
 * @file     edu_display.h
 * @brief    EDU mode screen: the times-table quiz on the LCD, run by the display thread.
 * @details  Replaces the transcript when the firmware is built with -DAPP_EDU_MODE=ON. The display
 *           thread hands it the recognized text and calls edu_display_poll() every
 *           EDU_DISPLAY_POLL_TICKS; it reads the touch screen (a Speak button) and the USER button,
 *           runs the quiz (edu_game.c) and redraws when something changed.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_EDU_EDU_DISPLAY_H
#define APPLICATION_EDU_EDU_DISPLAY_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>

#include "tx_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                                PUBLIC DEFINES                               */
/*******************************************************************************/

/** @brief Longest wait of the display thread between two polls (40 ms). */
#define EDU_DISPLAY_POLL_TICKS ((TX_TIMER_TICKS_PER_SECOND / 25U) + 1U)

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief Bring up the touch screen and the USER button, start the quiz, draw the first question.
 * @pre mw_lcd is initialized; call from the display thread.
 */
void edu_display_start(void);

/**
 * @brief Take recognized text; cheap (copies only), so it may run with the mailbox locked.
 * @param[in] final_text Newly committed characters, need not be terminated.
 * @param[in] final_length Characters in @p final_text.
 * @param[in] tentative Newest tentative tail, NUL-terminated.
 */
void edu_display_text(const char *final_text, uint32_t final_length, const char *tentative);

/**
 * @brief Read the inputs, run the quiz's timers, redraw if needed. Call from the display thread.
 */
void edu_display_poll(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_EDU_EDU_DISPLAY_H */
