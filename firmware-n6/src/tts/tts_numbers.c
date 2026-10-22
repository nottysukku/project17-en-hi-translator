/**
 * @file     tts_numbers.c
 * @brief    Numbers to Hindi words for the voice; implements tts_numbers.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tts_numbers.h"

#include <string.h>

/** @brief 0 to 99: each has its own word in Hindi. */
static const char *const words[100] = {
    "शून्य",   "एक",      "दो",      "तीन",     "चार",     "पाँच",    "छह",      "सात",     "आठ",       "नौ",
    "दस",      "ग्यारह",  "बारह",    "तेरह",    "चौदह",    "पंद्रह",  "सोलह",    "सत्रह",   "अठारह",    "उन्नीस",
    "बीस",     "इक्कीस",  "बाईस",    "तेईस",    "चौबीस",   "पच्चीस",  "छब्बीस",  "सत्ताईस", "अट्ठाईस",  "उनतीस",
    "तीस",     "इकतीस",   "बत्तीस",  "तैंतीस",  "चौंतीस",  "पैंतीस",  "छत्तीस",  "सैंतीस",  "अड़तीस",   "उनतालीस",
    "चालीस",   "इकतालीस", "बयालीस",  "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास",
    "पचास",    "इक्यावन", "बावन",    "तिरपन",   "चौवन",    "पचपन",    "छप्पन",   "सत्तावन", "अट्ठावन",  "उनसठ",
    "साठ",     "इकसठ",    "बासठ",    "तिरसठ",   "चौंसठ",   "पैंसठ",   "छियासठ",  "सड़सठ",   "अड़सठ",    "उनहत्तर",
    "सत्तर",   "इकहत्तर", "बहत्तर",  "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर",  "उन्यासी",
    "अस्सी",   "इक्यासी", "बयासी",   "तिरासी",  "चौरासी",  "पचासी",   "छियासी",  "सत्तासी", "अट्ठासी",  "नवासी",
    "नब्बे",   "इक्यानवे", "बानवे",  "तिरानवे", "चौरानवे", "पचानवे",  "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"};

/** @brief Longest run of digits read as one number (up to 99 crore); longer runs go digit by digit. */
#define TTS_NUMBER_DIGITS 9U

typedef struct
{
    char *out;
    uint32_t cap, n;
} writer_t;

static void put(writer_t *w, const char *s)
{
    while ((*s != '\0') && ((w->n + 1U) < w->cap))
    {
        w->out[w->n++] = *s++;
    }
}

/** @brief A word, with one space before it unless the output is empty or already ends in one. */
static void word(writer_t *w, const char *s)
{
    if ((w->n > 0U) && (w->out[w->n - 1U] != ' '))
    {
        put(w, " ");
    }
    put(w, s);
}

/** @brief The value of the digit at @p p (ASCII or Devanagari), -1 if none; *len its bytes. */
static int digit_at(const char *p, uint32_t *len)
{
    const uint8_t *u = (const uint8_t *)p;

    if ((u[0] >= '0') && (u[0] <= '9'))
    {
        *len = 1U;
        return u[0] - '0';
    }
    if ((u[0] == 0xE0U) && (u[1] == 0xA5U) && (u[2] >= 0xA6U) && (u[2] <= 0xAFU)) /* U+0966..U+096F */
    {
        *len = 3U;
        return u[2] - 0xA6U;
    }
    *len = 0U;
    return -1;
}

static void spell(writer_t *w, uint32_t n)
{
    static const struct
    {
        uint32_t unit, modulo;
        const char *name;
    } scales[] = {{100000U, 100U, "लाख"}, {1000U, 100U, "हज़ार"}, {100U, 10U, "सौ"}};
    uint32_t i;

    if (n == 0U)
    {
        word(w, words[0]);
        return;
    }
    if (n >= 10000000U)
    {
        spell(w, n / 10000000U);
        word(w, "करोड़");
    }
    for (i = 0; i < (sizeof(scales) / sizeof(scales[0])); ++i)
    {
        const uint32_t part = (n / scales[i].unit) % scales[i].modulo;
        if (part != 0U)
        {
            word(w, words[part]);
            word(w, scales[i].name);
        }
    }
    if ((n % 100U) != 0U)
    {
        word(w, words[n % 100U]);
    }
}

uint32_t tts_spell_numbers(const char *in, char *out, uint32_t cap)
{
    writer_t w = {out, cap, 0U};
    const char *p = in;

    if ((out == NULL) || (cap == 0U))
    {
        return 0U;
    }
    while ((p != NULL) && (*p != '\0') && ((w.n + 1U) < cap))
    {
        uint32_t len, digits = 0U, value = 0U;
        int d = digit_at(p, &len);
        const char *q = p, *start = p;

        if (d < 0)
        {
            out[w.n++] = *p++;
            continue;
        }
        /* the integer part: digits, and commas between digits (2,500 and 1,50,000) */
        while ((d = digit_at(q, &len)) >= 0)
        {
            uint32_t l2;
            value = (digits < TTS_NUMBER_DIGITS) ? ((value * 10U) + (uint32_t)d) : value;
            digits++;
            q += len;
            if ((*q == ',') && (digit_at(q + 1, &l2) >= 0))
            {
                q++;
            }
        }
        if (digits <= TTS_NUMBER_DIGITS)
        {
            spell(&w, value);
        }
        else
        {
            for (; start < q; start += (len > 0U) ? len : 1U) /* digit by digit */
            {
                d = digit_at(start, &len);
                if (d >= 0)
                {
                    word(&w, words[d]);
                }
            }
        }
        /* a decimal part, digit by digit */
        if ((*q == '.') && (digit_at(q + 1, &len) >= 0))
        {
            word(&w, "दशमलव");
            q++;
            while ((d = digit_at(q, &len)) >= 0)
            {
                word(&w, words[d]);
                q += len;
            }
        }
        if (*q == '%')
        {
            word(&w, "प्रतिशत");
            q++;
        }
        if ((*q != '\0') && (*q != ' '))
        {
            put(&w, " ");
        }
        p = q;
    }
    out[w.n] = '\0';
    return w.n;
}
