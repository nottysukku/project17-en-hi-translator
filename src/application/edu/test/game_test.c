/**
 * @file     game_test.c
 * @brief    Host test of the quiz flow (edu_game.c): turns played with a simulated clock.
 * @details  Text arrives the way the board delivers it (committed pieces and a tentative tail);
 *           the test checks when each turn is judged, the verdict, the messages and what comes next.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <string.h>

#include "edu_game.h"

static edu_game_t game;
static int failures;
static int checks;

#define CHECK(cond, ...)                                                                                     \
    do                                                                                                       \
    {                                                                                                        \
        ++checks;                                                                                            \
        if (!(cond))                                                                                         \
        {                                                                                                    \
            ++failures;                                                                                      \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                      \
            printf(__VA_ARGS__);                                                                             \
            printf("\n");                                                                                    \
        }                                                                                                    \
    } while (0)

/** @brief Run the clock from *now to until in 40 ms steps (the display thread's poll period). */
static void run(uint32_t *now, uint32_t until)
{
    while (*now < until)
    {
        *now += 40U;
        edu_game_tick(&game, *now);
    }
}

/** @brief Run until judged; returns the time it happened, or 0 if not by the limit. */
static uint32_t run_until_judged(uint32_t *now, uint32_t limit)
{
    while (*now < limit)
    {
        *now += 40U;
        edu_game_tick(&game, *now);
        if (game.phase == EDU_GAME_RESULT)
        {
            return *now;
        }
    }
    return 0U;
}

static void say(const char *committed, const char *tentative, uint32_t now)
{
    edu_game_text(&game, committed, (uint32_t)strlen(committed), tentative, now);
}

static void new_game(int16_t a, int16_t b)
{
    edu_game_init(&game, 1234U);
    game.question[0] = a;
    game.question[1] = b;
}

int main(void)
{
    uint32_t now = 1000U;
    uint32_t judged;

    /* 1. The answer only, right: judged 1.6 s after the last word, then a new question. */
    new_game(23, 8);
    CHECK(game.phase == EDU_GAME_ASK, "starts by asking");
    say("IGNORED", "", now); /* text before Speak is not part of the answer */
    edu_game_press(&game, now);
    CHECK(game.phase == EDU_GAME_LISTEN, "Speak starts listening");
    run(&now, 2000U);
    say("ONE EIGHTY", "FOUR", now);
    run(&now, 2400U);
    say(" FOUR.", "", now);
    judged = run_until_judged(&now, 20000U);
    CHECK(judged >= 2400U + EDU_GAME_SETTLE_MS && judged <= 2400U + EDU_GAME_SETTLE_MS + 40U,
          "judged 1.6 s after the last word (at %u)", (unsigned)judged);
    CHECK(game.verdict == EDU_GAME_RIGHT, "right (verdict %d, heard \"%s\")", game.verdict, game.heard);
    CHECK(strcmp(game.line1, "23 x 8 = 184") == 0, "line1 \"%s\"", game.line1);
    CHECK(game.right == 1U && game.streak == 1U, "score");
    run(&now, judged + EDU_GAME_SHOW_RIGHT_MS + 40U);
    CHECK(game.phase == EDU_GAME_ASK, "back to asking");
    CHECK(!(game.question[0] == 23 && game.question[1] == 8), "a new question");
    CHECK(game.question[0] >= EDU_GAME_TABLE_MIN && game.question[0] <= EDU_GAME_TABLE_MAX &&
              game.question[1] >= EDU_GAME_MULT_MIN && game.question[1] <= EDU_GAME_MULT_MAX,
          "question in range: %d x %d", game.question[0], game.question[1]);

    /* 2. The whole line, wrong: a cross, the right answer shown, streak reset. */
    new_game(23, 8);
    game.streak = 4U;
    edu_game_press(&game, now);
    run(&now, now + 500U);
    say("TWENTY THREE TIMES EIGHT IS ONE HUNDRED EIGHTY FIVE", "", now);
    {
        const uint32_t at = now;

        judged = run_until_judged(&now, now + 20000U);
        CHECK(judged - at >= EDU_GAME_SETTLE_MS && judged - at <= EDU_GAME_SETTLE_MS + 40U, "judged after %u ms",
              (unsigned)(judged - at));
    }
    CHECK(game.verdict == EDU_GAME_WRONG, "wrong (verdict %d)", game.verdict);
    CHECK(strcmp(game.line1, "You said 185.") == 0, "line1 \"%s\"", game.line1);
    CHECK(strcmp(game.line2, "23 x 8 = 184") == 0, "line2 \"%s\"", game.line2);
    CHECK(game.wrong == 1U && game.streak == 0U, "score after a cross");
    edu_game_press(&game, now); /* NEXT skips the wait */
    CHECK(game.phase == EDU_GAME_ASK && !(game.question[0] == 23 && game.question[1] == 8), "NEXT: new question");

    /* 3. A blurred -ty/-teen word: no tick, no cross, the same question again. */
    new_game(4, 15);
    edu_game_press(&game, now);
    run(&now, now + 300U);
    say("SIXTEE", "", now);
    judged = run_until_judged(&now, now + 20000U);
    CHECK(game.verdict == EDU_GAME_UNCLEAR, "unclear (verdict %d)", game.verdict);
    CHECK(strcmp(game.line1, "Did you say sixty or sixteen?") == 0, "line1 \"%s\"", game.line1);
    CHECK(game.right == 0U && game.wrong == 0U, "no score for a blurred word");
    run(&now, judged + EDU_GAME_SHOW_UNCLEAR_MS + 40U);
    CHECK(game.phase == EDU_GAME_ASK && game.question[0] == 4 && game.question[1] == 15, "same question again");

    /* 4. Nothing heard: give up after 9 s. */
    new_game(7, 7);
    edu_game_press(&game, now);
    {
        const uint32_t start = now;

        judged = run_until_judged(&now, now + 20000U);
        CHECK(game.verdict == EDU_GAME_NOTHING, "nothing (verdict %d)", game.verdict);
        CHECK(judged - start >= EDU_GAME_SILENCE_MS && judged - start <= EDU_GAME_SILENCE_MS + 40U,
              "gave up after %u ms", (unsigned)(judged - start));
    }

    /* 5. "ONE HUNDRED" may go on ("... AND FIVE"): wait 1.4 s longer. */
    new_game(25, 4);
    edu_game_press(&game, now);
    {
        const uint32_t at = now + 600U;

        run(&now, at);
        say("ONE HUNDRED", "", at);
        run(&now, at + EDU_GAME_SETTLE_MS + 200U);
        CHECK(game.phase == EDU_GAME_LISTEN, "still listening after \"ONE HUNDRED\"");
        judged = run_until_judged(&now, now + 20000U);
        CHECK(judged >= at + EDU_GAME_SETTLE_MS + EDU_GAME_MORE_MS, "waited for more (%u ms)", (unsigned)(judged - at));
        CHECK(game.verdict == EDU_GAME_RIGHT, "100 is right for 25 x 4 (verdict %d)", game.verdict);
    }

    /* 6. Unconfirmed words still coming in keep the turn open. */
    new_game(9, 9);
    edu_game_press(&game, now);
    {
        const uint32_t at = now + 400U;

        run(&now, at);
        say("EIGHTY", "", at); /* complete (80), but "EIGHTY" may go on */
        run(&now, at + 2000U);
        say("", "ONE", now);  /* the tentative tail shows the child is still speaking */
        run(&now, at + 3200U);
        CHECK(game.phase == EDU_GAME_LISTEN, "not judged while words are still arriving");
        say(" ONE", "", now);
        judged = run_until_judged(&now, now + 20000U);
        CHECK(game.verdict == EDU_GAME_RIGHT, "81 (verdict %d, heard \"%s\")", game.verdict, game.heard);
    }

    /* 7. Stop: a complete answer quiet for 1 s is judged 1 s after Stop. */
    new_game(6, 7);
    edu_game_press(&game, now);
    {
        const uint32_t at = now + 300U;

        run(&now, at);
        say("FORTY TWO", "", at);
        run(&now, at + 200U);
        edu_game_press(&game, now); /* STOP */
        CHECK(game.stopping, "stopping");
        judged = run_until_judged(&now, now + 20000U);
        CHECK(judged <= at + 200U + 1000U + 80U, "judged soon after Stop (%u ms)", (unsigned)(judged - at));
        CHECK(game.verdict == EDU_GAME_RIGHT, "42 (verdict %d)", game.verdict);
    }

    /* 8. Stop with no whole answer: judged after the grace period, "couldn't hear". */
    new_game(6, 7);
    edu_game_press(&game, now);
    run(&now, now + 120U);
    say("SIX TIMES SEVEN IS", "", now);
    run(&now, now + 120U);
    edu_game_press(&game, now);
    {
        const uint32_t stop = now;

        judged = run_until_judged(&now, now + 20000U);
        CHECK(game.verdict == EDU_GAME_UNCLEAR, "unclear (verdict %d)", game.verdict);
        CHECK(judged - stop >= EDU_GAME_STOP_GRACE_MS, "waited the grace period (%u ms)", (unsigned)(judged - stop));
    }

    /* 9. Speech that never stops is cut at 30 s. */
    new_game(3, 3);
    edu_game_press(&game, now);
    {
        const uint32_t start = now;
        int i;

        for (i = 0; i < 40; ++i)
        {
            run(&now, now + 800U);
            say(" BLAH", "", now);
        }
        CHECK(game.phase == EDU_GAME_RESULT, "cut at the turn limit");
        (void)start;
    }

    /* 10. The clock wraps around 2^32 ms without breaking the timers. */
    new_game(2, 5);
    now = 0xFFFFF000U;
    edu_game_press(&game, now);
    now += 120U;
    say("TEN", "", now);
    {
        uint32_t steps = 0U;

        while ((game.phase == EDU_GAME_LISTEN) && (steps < 1000U))
        {
            now += 40U;
            ++steps;
            edu_game_tick(&game, now);
        }
        CHECK(game.verdict == EDU_GAME_RIGHT && steps * 40U >= EDU_GAME_SETTLE_MS && steps * 40U < 3000U,
              "judged across the wrap after %u ms", (unsigned)(steps * 40U));
    }

    printf("%d of %d game checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
