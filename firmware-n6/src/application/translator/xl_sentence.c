/**
 * @file     xl_sentence.c
 * @brief    Sentence cutting of the recognized English; implements xl_sentence.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xl_sentence.h"

#include <string.h>

/** @brief Pending committed text: room for two sentences. */
#define XL_SENTENCE_PENDING (2U * XL_SENTENCE_MAX_CHARS)

static char pending[XL_SENTENCE_PENDING + 1U];
static uint32_t pending_len;
static uint32_t last_growth_ms;
static int tentative_empty = 1;

static int is_letter(char c)
{
    return ((c >= 'A') && (c <= 'Z')) || ((c >= 'a') && (c <= 'z'));
}

static int is_end_mark(char c)
{
    return (c == '.') || (c == '?') || (c == '!');
}

/** @brief A period after one of these is part of the word, not the end of the sentence. */
static int is_abbreviation(const char *text, uint32_t dot)
{
    static const char *const words[] = {"MR", "MRS", "MS", "DR", "ST", "JR", "SR", "PROF", "MT", "NO", "VS"};
    uint32_t start = dot, i;

    while ((start > 0U) && is_letter(text[start - 1U]))
    {
        start--;
    }
    for (i = 0; i < (sizeof(words) / sizeof(words[0])); ++i)
    {
        const uint32_t n = (uint32_t)strlen(words[i]);
        uint32_t k;

        if ((dot - start) != n)
        {
            continue;
        }
        for (k = 0; k < n; ++k)
        {
            const char c = text[start + k];
            if (((c >= 'a') && (c <= 'z') ? (char)(c - 32) : c) != words[i][k])
            {
                break;
            }
        }
        if (k == n)
        {
            return 1;
        }
    }
    return 0;
}

/** @brief Drop the first @p n pending characters. */
static void consume(uint32_t n)
{
    if (n > pending_len)
    {
        n = pending_len;
    }
    (void)memmove(pending, &pending[n], pending_len - n);
    pending_len -= n;
    pending[pending_len] = '\0';
}

void xl_sentence_reset(void)
{
    pending_len = 0U;
    pending[0] = '\0';
    tentative_empty = 1;
}

void xl_sentence_feed(const char *committed, uint32_t length, const char *tentative, uint32_t now_ms)
{
    uint32_t i;

    if ((committed != NULL) && (length > 0U))
    {
        if (length > XL_SENTENCE_PENDING)
        {
            committed += length - XL_SENTENCE_PENDING;
            length = XL_SENTENCE_PENDING;
        }
        if ((pending_len + length) > XL_SENTENCE_PENDING)
        {
            consume((pending_len + length) - XL_SENTENCE_PENDING); /* not taken in time: oldest go */
        }
        (void)memcpy(&pending[pending_len], committed, length);
        pending_len += length;
        pending[pending_len] = '\0';
        for (i = 0; i < length; ++i)
        {
            if (committed[i] != ' ')
            {
                last_growth_ms = now_ms;
                break;
            }
        }
    }
    tentative_empty = 1;
    if (tentative != NULL)
    {
        for (i = 0; tentative[i] != '\0'; ++i)
        {
            if (tentative[i] != ' ')
            {
                tentative_empty = 0;
                last_growth_ms = now_ms; /* still talking */
                break;
            }
        }
    }
}

uint32_t xl_sentence_take(char *out, uint32_t capacity, uint32_t now_ms)
{
    uint32_t end = 0U, i, n, letters;

    if ((out == NULL) || (capacity < 2U))
    {
        return 0U;
    }
    out[0] = '\0';
    for (;;)
    {
        /* leading spaces never start a sentence */
        for (i = 0; (i < pending_len) && (pending[i] == ' '); ++i)
        {
        }
        consume(i);
        if (pending_len == 0U)
        {
            return 0U;
        }
        end = 0U;
        for (i = 0; i < pending_len; ++i)
        {
            if (is_end_mark(pending[i]) && !((pending[i] == '.') && is_abbreviation(pending, i)))
            {
                end = i + 1U;
                break;
            }
        }
        if ((end == 0U) && (pending_len >= XL_SENTENCE_MAX_CHARS))
        {
            /* too long: cut after the last comma, else at the last space, else hard */
            uint32_t comma = 0U, space = 0U;
            for (i = 0; i < XL_SENTENCE_MAX_CHARS; ++i)
            {
                if (pending[i] == ',')
                {
                    comma = i + 1U;
                }
                else if (pending[i] == ' ')
                {
                    space = i;
                }
            }
            end = (comma > (XL_SENTENCE_MAX_CHARS / 2U)) ? comma : ((space > 0U) ? space : XL_SENTENCE_MAX_CHARS);
        }
        if ((end == 0U) && tentative_empty && ((uint32_t)(now_ms - last_growth_ms) >= XL_SENTENCE_PAUSE_MS))
        {
            end = pending_len;
        }
        if (end == 0U)
        {
            return 0U;
        }
        letters = 0U;
        for (i = 0; i < end; ++i)
        {
            letters += (uint32_t)is_letter(pending[i]);
        }
        if (letters > 0U)
        {
            break;
        }
        consume(end); /* only marks or spaces: nothing to translate */
    }
    n = end;
    while ((n > 0U) && (pending[n - 1U] == ' '))
    {
        n--;
    }
    if (n >= capacity)
    {
        n = capacity - 1U;
    }
    (void)memcpy(out, pending, n);
    out[n] = '\0';
    consume(end);
    return n;
}
