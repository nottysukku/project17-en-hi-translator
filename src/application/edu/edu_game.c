/**
 * @file     edu_game.c
 * @brief    The times-table quiz; implements edu_game.h (the rules of ASR_Math_Game/app.py, quiz mode).
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "edu_game.h"

#include <stddef.h>
#include <string.h>

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief After Stop, a complete answer quiet this long is judged at once. */
#define EDU_GAME_STOP_QUIET_MS 1000U

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Milliseconds from since to now on a wrapping clock; 0 if since is (slightly) ahead. */
static uint32_t elapsed(uint32_t now, uint32_t since)
{
    const int32_t d = (int32_t)(now - since);

    return (d > 0) ? (uint32_t)d : 0U;
}

static uint32_t next_random(edu_game_t *game)
{
    uint32_t x = game->rng;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    game->rng = (x != 0U) ? x : 0x9E3779B9U;
    return game->rng;
}

static int has_letter(const char *text, uint32_t length)
{
    uint32_t i;

    for (i = 0U; i < length; ++i)
    {
        const char c = text[i];

        if (((c >= 'A') && (c <= 'Z')) || ((c >= 'a') && (c <= 'z')))
        {
            return 1;
        }
    }
    return 0;
}

/** @brief Bounded string builder for the message lines. */
static void put_text(char *line, const char *text)
{
    uint32_t used = (uint32_t)strlen(line);

    while ((*text != '\0') && (used + 1U < EDU_GAME_LINE_CAPACITY))
    {
        line[used++] = *text++;
    }
    line[used] = '\0';
}

static void put_number(char *line, int32_t value)
{
    char digits[12];
    char text[12];
    uint32_t count = 0U;
    uint32_t k = 0U;

    if (value < 0)
    {
        put_text(line, "?");
        return;
    }
    do
    {
        digits[count++] = (char)('0' + (value % 10));
        value /= 10;
    } while ((value > 0) && (count < sizeof(digits) - 1U));
    while (count > 0U)
    {
        text[k++] = digits[--count];
    }
    text[k] = '\0';
    put_text(line, text);
}

/** @brief "SIXTY" -> "sixty". */
static void put_lower(char *line, const char *word)
{
    char text[16];
    uint32_t i;

    for (i = 0U; (word[i] != '\0') && (i + 1U < sizeof(text)); ++i)
    {
        text[i] = ((word[i] >= 'A') && (word[i] <= 'Z')) ? (char)(word[i] - 'A' + 'a') : word[i];
    }
    text[i] = '\0';
    put_text(line, text);
}

/** @brief A new question, never the one just asked (app.py next_question). */
static void new_question(edu_game_t *game)
{
    const uint32_t tables = (uint32_t)(EDU_GAME_TABLE_MAX - EDU_GAME_TABLE_MIN + 1);
    const uint32_t mults = (uint32_t)(EDU_GAME_MULT_MAX - EDU_GAME_MULT_MIN + 1);
    int16_t a;
    int16_t b;
    uint32_t tries = 0U;

    do
    {
        a = (int16_t)(EDU_GAME_TABLE_MIN + (int32_t)(next_random(game) % tables));
        b = (int16_t)(EDU_GAME_MULT_MIN + (int32_t)(next_random(game) % mults));
        ++tries;
    } while ((a == game->question[0]) && (b == game->question[1]) && (tries < 8U));
    game->question[0] = a;
    game->question[1] = b;
}

static void clear_turn(edu_game_t *game)
{
    game->heard[0] = '\0';
    game->heard_length = 0U;
    game->tentative[0] = '\0';
    game->live[0] = '\0';
    game->line1[0] = '\0';
    game->line2[0] = '\0';
    game->verdict = EDU_GAME_VERDICT_NONE;
    game->stopping = 0U;
    game->text_changed = 0U;
    game->complete = 0U;
    game->may_continue = 0U;
}

static void ask(edu_game_t *game)
{
    clear_turn(game);
    game->phase = EDU_GAME_ASK;
    ++game->revision;
}

/** @brief Read the turn's text again: the live reading and whether it is a whole answer. */
static void reread(edu_game_t *game)
{
    const edu_math_claim_t *claim = &game->result.claim;

    (void)edu_math_parse(&game->work, game->heard, game->question, NULL, &game->result);
    /* quiz mode: any reading with an answer in it is complete (app.py tick) */
    game->complete = (uint8_t)(claim->c != EDU_MATH_NONE);
    game->may_continue = (uint8_t)edu_math_may_continue(&game->work, game->heard);
    game->live[0] = '\0';
    if (game->result.doubt)
    {
        put_text(game->live, game->result.shown);
    }
    else if (claim->template_id != EDU_MATH_TEMPLATE_NONE)
    {
        edu_math_format(claim, game->live, EDU_GAME_LINE_CAPACITY);
    }
}

/** @brief Mark the turn and say why (app.py _judge). */
static void judge(edu_game_t *game, uint32_t now_ms)
{
    const edu_math_claim_t *claim = &game->result.claim;
    uint32_t show_ms = EDU_GAME_SHOW_UNCLEAR_MS;

    game->line1[0] = '\0';
    game->line2[0] = '\0';
    if (!has_letter(game->heard, game->heard_length))
    {
        game->verdict = EDU_GAME_NOTHING;
        put_text(game->line1, "I didn't hear anything.");
        put_text(game->line2, "Press SPEAK and try again.");
    }
    else if (claim->correct == 1)
    {
        game->verdict = EDU_GAME_RIGHT;
        edu_math_format(claim, game->line1, EDU_GAME_LINE_CAPACITY);
        put_text(game->line2, "Well done!");
        ++game->right;
        ++game->streak;
        if (game->streak > game->best_streak)
        {
            game->best_streak = game->streak;
        }
        show_ms = EDU_GAME_SHOW_RIGHT_MS;
    }
    else if (claim->correct == 0)
    {
        game->verdict = EDU_GAME_WRONG;
        put_text(game->line1, "You said ");
        put_number(game->line1, claim->c);
        put_text(game->line1, ".");
        put_number(game->line2, game->question[0]);
        put_text(game->line2, " x ");
        put_number(game->line2, game->question[1]);
        put_text(game->line2, " = ");
        put_number(game->line2, (int32_t)game->question[0] * game->question[1]);
        ++game->wrong;
        game->streak = 0U;
        show_ms = EDU_GAME_SHOW_WRONG_MS;
    }
    else if (game->result.doubt)
    {
        uint32_t i;

        /* a blurred -ty / -teen word: neither a tick nor a cross is earned from it */
        game->verdict = EDU_GAME_UNCLEAR;
        put_text(game->line1, "Did you say ");
        for (i = 0U; i < game->result.blurred; ++i)
        {
            if (i > 0U)
            {
                put_text(game->line1, " / ");
            }
            put_lower(game->line1, game->result.blur_ty[i]);
            put_text(game->line1, " or ");
            put_lower(game->line1, game->result.blur_teen[i]);
        }
        put_text(game->line1, "?");
        put_text(game->line2, "Say it again, clearly.");
    }
    else
    {
        game->verdict = EDU_GAME_UNCLEAR;
        put_text(game->line1, "I couldn't hear a whole answer.");
        put_text(game->line2, "Press SPEAK and say it again.");
    }
    game->phase = EDU_GAME_RESULT;
    game->result_until = now_ms + show_ms;
    ++game->revision;
}

/** @brief After the verdict: a new question if it was marked, else the same one again. */
static void move_on(edu_game_t *game)
{
    if ((game->verdict == EDU_GAME_RIGHT) || (game->verdict == EDU_GAME_WRONG))
    {
        new_question(game);
    }
    ask(game);
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void edu_game_init(edu_game_t *game, uint32_t seed)
{
    (void)memset(game, 0, sizeof(*game));
    game->rng = (seed != 0U) ? seed : 0x9E3779B9U;
    new_question(game);
    ask(game);
}

void edu_game_press(edu_game_t *game, uint32_t now_ms)
{
    game->rng ^= now_ms * 2654435761U; /* when a child presses is random enough */
    switch (game->phase)
    {
    case EDU_GAME_ASK:
        clear_turn(game);
        game->phase = EDU_GAME_LISTEN;
        game->turn_start = now_ms;
        game->last_heard = now_ms;
        (void)edu_math_parse(&game->work, "", game->question, NULL, &game->result);
        ++game->revision;
        break;
    case EDU_GAME_LISTEN:
        if (!game->stopping)
        {
            game->stopping = 1U;
            game->stopped_at = now_ms;
            ++game->revision;
        }
        break;
    default:
        move_on(game);
        break;
    }
}

void edu_game_text(edu_game_t *game, const char *final_text, uint32_t final_length, const char *tentative,
                   uint32_t now_ms)
{
    if (game->phase != EDU_GAME_LISTEN)
    {
        return;
    }
    if ((final_text != NULL) && (final_length > 0U))
    {
        uint32_t room = EDU_GAME_HEARD_CAPACITY - 1U - game->heard_length;
        uint32_t n = (final_length < room) ? final_length : room;

        (void)memcpy(&game->heard[game->heard_length], final_text, n);
        game->heard_length += n;
        game->heard[game->heard_length] = '\0';
        if (has_letter(final_text, final_length))
        {
            game->last_heard = now_ms;
        }
        game->text_changed = 1U;
        ++game->revision;
    }
    if ((tentative != NULL) && (strncmp(tentative, game->tentative, EDU_GAME_TENTATIVE_CAPACITY) != 0))
    {
        (void)strncpy(game->tentative, tentative, EDU_GAME_TENTATIVE_CAPACITY - 1U);
        game->tentative[EDU_GAME_TENTATIVE_CAPACITY - 1U] = '\0';
        /* words still being recognized: the child is speaking */
        if (has_letter(game->tentative, (uint32_t)strlen(game->tentative)))
        {
            game->last_heard = now_ms;
        }
        ++game->revision;
    }
}

void edu_game_tick(edu_game_t *game, uint32_t now_ms)
{
    if (game->phase == EDU_GAME_LISTEN)
    {
        const uint32_t quiet = elapsed(now_ms, game->last_heard);
        const uint32_t turn = elapsed(now_ms, game->turn_start);
        uint32_t settle;
        int done;

        if (game->text_changed)
        {
            game->text_changed = 0U;
            reread(game);
        }
        settle = EDU_GAME_SETTLE_MS + (game->may_continue ? EDU_GAME_MORE_MS : 0U);
        done = game->complete && (quiet >= settle);
        if (game->stopping)
        {
            const uint32_t since = elapsed(now_ms, game->stopped_at);

            done = done || (since >= EDU_GAME_STOP_GRACE_MS) ||
                   (game->complete && (since >= EDU_GAME_STOP_QUIET_MS) && (quiet >= EDU_GAME_STOP_QUIET_MS));
        }
        if (!has_letter(game->heard, game->heard_length) && (turn >= EDU_GAME_SILENCE_MS))
        {
            done = 1;
        }
        if (turn >= EDU_GAME_TURN_MAX_MS)
        {
            done = 1;
        }
        if (done)
        {
            judge(game, now_ms);
        }
    }
    else if (game->phase == EDU_GAME_RESULT)
    {
        if ((int32_t)(now_ms - game->result_until) >= 0)
        {
            move_on(game);
        }
    }
}
