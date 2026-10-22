/**
 * @file     edu_game.h
 * @brief    The times-table quiz on the board: ask, listen, mark with a tick or a cross, next question.
 * @details  The rules of ASR_Math_Game/app.py (quiz mode) without the web page: press Speak, say the
 *           answer ("one eighty four") or the whole line ("twenty three times eight is one hundred
 *           eighty four"); the turn is judged once the words form an answer and the recognizer has
 *           been quiet a moment, or shortly after Speak is pressed again.
 *
 *           Plain C, no RTOS and no drawing: the caller passes the time and the recognized text and
 *           draws the state (edu_display.c on the board; a host test can drive it the same way).
 *           Not thread safe: one thread calls everything.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_EDU_EDU_GAME_H
#define APPLICATION_EDU_EDU_GAME_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>

#include "edu_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                                PUBLIC DEFINES                               */
/*******************************************************************************/

/** @brief Questions: EDU_GAME_TABLE_MIN..MAX times EDU_GAME_MULT_MIN..MAX (app.py defaults). */
#ifndef EDU_GAME_TABLE_MIN
#define EDU_GAME_TABLE_MIN 2
#endif
#ifndef EDU_GAME_TABLE_MAX
#define EDU_GAME_TABLE_MAX 30
#endif
#ifndef EDU_GAME_MULT_MIN
#define EDU_GAME_MULT_MIN 1
#endif
#ifndef EDU_GAME_MULT_MAX
#define EDU_GAME_MULT_MAX 15
#endif

/** @brief No new text for this long after a complete answer: judge it (app.py SETTLE_S). */
#define EDU_GAME_SETTLE_MS 1600U
/** @brief ... plus this when the last word expects more: "ONE HUNDRED", "EIGHTY", "IS" (MORE_S). */
#define EDU_GAME_MORE_MS 1400U
/** @brief After Stop, wait this long for the last words; the board commits about 1.8 s late (STOP_GRACE_S). */
#define EDU_GAME_STOP_GRACE_MS 2600U
/** @brief Nothing heard for this long: give up (SILENCE_S). */
#define EDU_GAME_SILENCE_MS 9000U
/** @brief Longest turn, whatever is heard. */
#define EDU_GAME_TURN_MAX_MS 30000U
/** @brief How long the tick, the cross or the "say it again" stays before the game moves on. */
#define EDU_GAME_SHOW_RIGHT_MS 2500U
#define EDU_GAME_SHOW_WRONG_MS 4500U
#define EDU_GAME_SHOW_UNCLEAR_MS 3500U

/** @brief Text of one turn kept for checking, terminator included; later words are dropped. */
#define EDU_GAME_HEARD_CAPACITY 256U
/** @brief Room for the recognizer's tentative tail. */
#define EDU_GAME_TENTATIVE_CAPACITY 128U
/** @brief Room for each message line. */
#define EDU_GAME_LINE_CAPACITY 64U

/*******************************************************************************/
/*                                 PUBLIC TYPES                                */
/*******************************************************************************/

/** @brief What the screen shows. */
typedef enum
{
    EDU_GAME_ASK = 0, /**< The question and a Speak button. */
    EDU_GAME_LISTEN,  /**< Listening: the words heard so far, a Stop button. */
    EDU_GAME_RESULT   /**< The verdict: tick, cross, or "say it again". */
} edu_game_phase_t;

/** @brief The verdict of a turn. */
typedef enum
{
    EDU_GAME_VERDICT_NONE = 0, /**< Not judged yet. */
    EDU_GAME_RIGHT,            /**< A tick. */
    EDU_GAME_WRONG,            /**< A cross. */
    EDU_GAME_UNCLEAR,          /**< No tick and no cross: a blurred word or no whole answer; ask again. */
    EDU_GAME_NOTHING           /**< Nothing heard. */
} edu_game_verdict_t;

/** @brief The whole game: read it to draw the screen. */
typedef struct
{
    edu_game_phase_t phase;      /**< What to draw. */
    edu_game_verdict_t verdict;  /**< In EDU_GAME_RESULT: what to draw. */
    int16_t question[2];         /**< The question on screen, a x b. */
    char heard[EDU_GAME_HEARD_CAPACITY];         /**< Final text of this turn. */
    char tentative[EDU_GAME_TENTATIVE_CAPACITY]; /**< The recognizer's newest, unconfirmed words. */
    char live[EDU_GAME_LINE_CAPACITY];           /**< While listening: what the words read as, "23 x 8 = 184". */
    char line1[EDU_GAME_LINE_CAPACITY];          /**< In EDU_GAME_RESULT: the main message. */
    char line2[EDU_GAME_LINE_CAPACITY];          /**< In EDU_GAME_RESULT: the second line. */
    uint16_t right;              /**< Score: answers ticked. */
    uint16_t wrong;              /**< Score: answers crossed. */
    uint16_t streak;             /**< Ticks in a row. */
    uint16_t best_streak;        /**< Longest streak. */
    uint32_t revision;           /**< Changes whenever the screen should be redrawn. */

    /* private */
    uint32_t rng;
    uint32_t turn_start;
    uint32_t last_heard;
    uint32_t stopped_at;
    uint32_t result_until;
    uint8_t stopping;
    uint8_t text_changed;
    uint8_t complete;
    uint8_t may_continue;
    uint32_t heard_length;
    edu_math_result_t result;
    edu_math_work_t work;
} edu_game_t;

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief Start a game: score zero, first question.
 * @param[out] game The game.
 * @param[in] seed Any number; varies the questions.
 */
void edu_game_init(edu_game_t *game, uint32_t seed);

/**
 * @brief The Speak / Stop / Next button was pressed.
 * @param[in,out] game The game.
 * @param[in] now_ms A millisecond clock (wraps safely).
 */
void edu_game_press(edu_game_t *game, uint32_t now_ms);

/**
 * @brief New text from the recognizer; ignored unless listening. Cheap: only copies.
 * @param[in,out] game The game.
 * @param[in] final_text Newly committed characters (need not be terminated).
 * @param[in] final_length Characters in @p final_text.
 * @param[in] tentative The newest tentative tail, NUL-terminated, or NULL.
 * @param[in] now_ms The clock.
 */
void edu_game_text(edu_game_t *game, const char *final_text, uint32_t final_length, const char *tentative,
                   uint32_t now_ms);

/**
 * @brief Re-read the text and run the timers: judges the turn, moves to the next question.
 * @details Call every few tens of milliseconds and after edu_game_text().
 * @param[in,out] game The game.
 * @param[in] now_ms The clock.
 */
void edu_game_tick(edu_game_t *game, uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_EDU_EDU_GAME_H */
