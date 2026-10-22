/**
 * @file     sentence_host.c
 * @brief    PC test of xl_sentence.c: scripted recognizer output in, sentences out.
 *   cc -I.. ../xl_sentence.c sentence_host.c -o sentence_host && ./sentence_host
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <string.h>

#include "xl_sentence.h"

static int failures;

/** @brief One publish (every 500 ms on the board), then take everything ready. */
static void step(uint32_t t, const char *committed, const char *tentative, const char *expect1, const char *expect2)
{
    char out[XL_SENTENCE_CAPACITY];
    const char *expect[2] = {expect1, expect2};
    int k;

    xl_sentence_feed(committed, (uint32_t)strlen(committed), tentative, t);
    for (k = 0; k < 2; ++k)
    {
        const uint32_t n = xl_sentence_take(out, sizeof(out), t);
        const char *want = expect[k] ? expect[k] : "";
        if (strcmp(out, want) != 0 || (n != strlen(want)))
        {
            printf("FAIL at %u ms: got \"%s\", expected \"%s\"\n", t, out, want);
            failures++;
        }
        if (n == 0U)
        {
            break;
        }
    }
}

int main(void)
{
    char big[300];
    int i;

    /* a pause ends a sentence without a mark */
    xl_sentence_reset();
    step(0, "", "HELLO", NULL, NULL);
    step(500, "HELLO HOW", " ARE", NULL, NULL);
    step(1000, " ARE YOU", "", NULL, NULL);
    step(1500, "", "", NULL, NULL);
    step(2000, "", "", NULL, NULL);
    step(2500, "", "", "HELLO HOW ARE YOU", NULL);
    step(3000, "", "", NULL, NULL);

    /* end marks, two sentences in one publish, abbreviations */
    xl_sentence_reset();
    step(10000, "MR. SHARMA IS HERE. WHERE IS", " THE", "MR. SHARMA IS HERE.", NULL);
    step(10500, " THE STATION?", "", "WHERE IS THE STATION?", NULL);

    /* talking keeps a sentence open even when nothing is committed for a while */
    xl_sentence_reset();
    step(20000, "I THINK", " THAT", NULL, NULL);
    step(20500, "", " THAT WE", NULL, NULL);
    step(21000, "", " THAT WE SHOULD", NULL, NULL);
    step(21500, "", " THAT WE SHOULD", NULL, NULL);
    step(22000, " THAT WE SHOULD GO", "", NULL, NULL);
    step(23200, "", "", "I THINK THAT WE SHOULD GO", NULL);

    /* marks alone are dropped */
    xl_sentence_reset();
    step(30000, " . ,", "", NULL, NULL);
    step(32000, "", "", NULL, NULL);

    /* very long speech without pauses is cut at a comma or a space */
    xl_sentence_reset();
    big[0] = '\0';
    for (i = 0; i < 40; ++i)
    {
        strcat(big, i == 25 ? "WORDS, " : "WORDS ");
    }
    xl_sentence_feed(big, (uint32_t)strlen(big), "MORE", 40000);
    {
        char out[XL_SENTENCE_CAPACITY];
        const uint32_t n = xl_sentence_take(out, sizeof(out), 40000);
        if (n == 0U || n > XL_SENTENCE_MAX_CHARS || out[n - 1U] == ' ')
        {
            printf("FAIL: long text cut as \"%s\"\n", out);
            failures++;
        }
        else
        {
            printf("long text cut after %u characters: ...%s\n", n, out + n - 20);
        }
    }

    printf(failures ? "%d FAILURES\n" : "all sentence tests passed\n", failures);
    return failures != 0;
}
