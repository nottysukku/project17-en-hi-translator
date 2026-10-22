/**
 * @file     xl_hindi_text.c
 * @brief    Devanagari text from a pre-shaped syllable font; implements xl_hindi_text.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xl_hindi_text.h"

#include <stddef.h>
#include <string.h>

#ifndef XL_HINDI_TEXT_HOST_TEST
#include "mw_lcd.h"
#else
void mw_lcd_draw_alpha4(int32_t x, int32_t y, uint32_t width, uint32_t height, const uint8_t *bits, uint16_t color);
#endif

/** @brief Font header, as written by hi_font_build.py. */
typedef struct
{
    char magic[4]; /* "HIFT" */
    uint32_t version, px, ascent, descent, line_height, count, index_off, strings_off, bitmaps_off;
} hift_header_t;

/** @brief One syllable of the font. */
typedef struct
{
    uint32_t str_off, bmp_off;
    uint16_t str_len, advance;
    uint8_t width, height;
    int8_t x_off, y_off;
} hift_entry_t;

static const uint8_t *font;
static const hift_header_t *hdr;

/* ------------------------------------------------------------------------------------------ */
/*                                       UTF-8 and syllables                                  */
/* ------------------------------------------------------------------------------------------ */

/** @brief Decode one code point; *len gets its byte length (0 at the end of the string). */
static uint32_t decode(const char *s, uint32_t *len)
{
    const uint8_t *p = (const uint8_t *)s;

    if (p[0] == 0U)
    {
        *len = 0;
        return 0;
    }
    if (p[0] < 0x80U)
    {
        *len = 1;
        return p[0];
    }
    if (((p[0] & 0xE0U) == 0xC0U) && (p[1] != 0U))
    {
        *len = 2;
        return ((uint32_t)(p[0] & 0x1FU) << 6) | (p[1] & 0x3FU);
    }
    if (((p[0] & 0xF0U) == 0xE0U) && (p[1] != 0U) && (p[2] != 0U))
    {
        *len = 3;
        return ((uint32_t)(p[0] & 0x0FU) << 12) | ((uint32_t)(p[1] & 0x3FU) << 6) | (p[2] & 0x3FU);
    }
    if (((p[0] & 0xF8U) == 0xF0U) && (p[1] != 0U) && (p[2] != 0U) && (p[3] != 0U))
    {
        *len = 4;
        return ((uint32_t)(p[0] & 0x07U) << 18) | ((uint32_t)(p[1] & 0x3FU) << 12) | ((uint32_t)(p[2] & 0x3FU) << 6) |
               (p[3] & 0x3FU);
    }
    *len = 1; /* invalid byte: skip it */
    return 0xFFFDU;
}

static int is_cons(uint32_t c)
{
    return ((c >= 0x0915U) && (c <= 0x0939U)) || ((c >= 0x0958U) && (c <= 0x095FU)) ||
           ((c >= 0x0978U) && (c <= 0x097FU));
}

static int is_vowel(uint32_t c)
{
    return ((c >= 0x0904U) && (c <= 0x0914U)) || (c == 0x0960U) || (c == 0x0961U) || ((c >= 0x0972U) && (c <= 0x0977U));
}

static int is_matra(uint32_t c)
{
    return ((c >= 0x093AU) && (c <= 0x093BU)) || ((c >= 0x093EU) && (c <= 0x094CU)) || (c == 0x094EU) ||
           (c == 0x094FU) || ((c >= 0x0955U) && (c <= 0x0957U)) || (c == 0x0962U) || (c == 0x0963U);
}

static int is_mod(uint32_t c)
{
    return (c >= 0x0900U) && (c <= 0x0903U);
}

#define NUKTA  0x093CU
#define HALANT 0x094DU

uint32_t xl_hindi_cluster_len(const char *s)
{
    uint32_t n, l, l2, l3;
    uint32_t c = decode(s, &n);

    if (n == 0U)
    {
        return 0;
    }
    if (is_cons(c) || is_vowel(c))
    {
        if (is_cons(c))
        {
            for (;;)
            {
                if (decode(s + n, &l) == NUKTA)
                {
                    n += l;
                }
                if (decode(s + n, &l) == HALANT)
                {
                    uint32_t k = n + l;
                    const uint32_t z = decode(s + k, &l2);
                    if ((z == 0x200CU) || (z == 0x200DU))
                    {
                        k += l2;
                    }
                    if (is_cons(decode(s + k, &l3)))
                    {
                        n = k + l3;
                        continue;
                    }
                    n = k; /* final halant (and ZWJ/ZWNJ) stays in the syllable */
                }
                break;
            }
        }
        else if (decode(s + n, &l) == NUKTA)
        {
            n += l;
        }
        while (is_matra(decode(s + n, &l)))
        {
            n += l;
        }
        if (decode(s + n, &l) == NUKTA)
        {
            n += l;
        }
        while (is_mod(decode(s + n, &l)))
        {
            n += l;
        }
    }
    else if (is_matra(c) || is_mod(c) || (c == NUKTA) || (c == HALANT))
    {
        uint32_t d;
        while (((d = decode(s + n, &l)) != 0U) && (is_matra(d) || is_mod(d)))
        {
            n += l;
        }
    }
    return n;
}

/* ------------------------------------------------------------------------------------------ */
/*                                          Lookup                                            */
/* ------------------------------------------------------------------------------------------ */

static const hift_entry_t *find(const char *s, uint32_t len)
{
    const hift_entry_t *idx = (const hift_entry_t *)(const void *)(font + hdr->index_off);
    const uint8_t *strings = font + hdr->strings_off;
    int32_t lo = 0, hi = (int32_t)hdr->count - 1;

    while (lo <= hi)
    {
        const int32_t mid = (lo + hi) / 2;
        const hift_entry_t *e = &idx[mid];
        const uint32_t common = (e->str_len < len) ? e->str_len : len;
        int cmp = memcmp(strings + e->str_off, s, common);

        if (cmp == 0)
        {
            cmp = (e->str_len < len) ? -1 : ((e->str_len > len) ? 1 : 0);
        }
        if (cmp == 0)
        {
            return e;
        }
        if (cmp < 0)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid - 1;
        }
    }
    return NULL;
}

/** @brief Draw (if draw) and measure one syllable; letter by letter if it is not in the font. */
static uint32_t syllable(int32_t x, int32_t baseline, const char *s, uint32_t len, int draw, uint16_t color)
{
    const hift_entry_t *e = find(s, len);
    uint32_t adv = 0, i = 0, l;

    if (e != NULL)
    {
        if (draw && (e->width > 0U))
        {
            mw_lcd_draw_alpha4(x + e->x_off, baseline + e->y_off, e->width, e->height,
                               font + hdr->bitmaps_off + e->bmp_off, color);
        }
        return e->advance;
    }
    while (i < len)
    {
        (void)decode(s + i, &l);
        if (l == 0U)
        {
            break;
        }
        e = find(s + i, l);
        if (e != NULL)
        {
            if (draw && (e->width > 0U))
            {
                mw_lcd_draw_alpha4(x + (int32_t)adv + e->x_off, baseline + e->y_off, e->width, e->height,
                                   font + hdr->bitmaps_off + e->bmp_off, color);
            }
            adv += e->advance;
        }
        i += l;
    }
    return adv;
}

/** @brief Width of (and optionally draw) the first n bytes of s on one line. */
static uint32_t run(int32_t x, int32_t baseline, const char *s, uint32_t n, int draw, uint16_t color)
{
    uint32_t i = 0, w = 0;

    while (i < n)
    {
        uint32_t len = xl_hindi_cluster_len(s + i);
        if (len == 0U)
        {
            break;
        }
        if ((i + len) > n)
        {
            len = n - i;
        }
        w += syllable(x + (int32_t)w, baseline, s + i, len, draw, color);
        i += len;
    }
    return w;
}

/* ------------------------------------------------------------------------------------------ */
/*                                        Public API                                          */
/* ------------------------------------------------------------------------------------------ */

int xl_hindi_text_open(const void *blob)
{
    const hift_header_t *h = (const hift_header_t *)blob;

    if ((h == NULL) || (memcmp(h->magic, "HIFT", 4) != 0) || (h->version != 1U))
    {
        font = NULL;
        hdr = NULL;
        return -1;
    }
    font = (const uint8_t *)blob;
    hdr = h;
    return 0;
}

uint32_t xl_hindi_text_line_height(void)
{
    return (hdr != NULL) ? hdr->line_height : 0U;
}

uint32_t xl_hindi_text_ascent(void)
{
    return (hdr != NULL) ? hdr->ascent : 0U;
}

uint32_t xl_hindi_text_width(const char *utf8)
{
    return ((hdr != NULL) && (utf8 != NULL)) ? run(0, 0, utf8, (uint32_t)strlen(utf8), 0, 0) : 0U;
}

void xl_hindi_text_draw(int32_t x, int32_t baseline, const char *utf8, uint16_t color)
{
    if ((hdr != NULL) && (utf8 != NULL))
    {
        (void)run(x, baseline, utf8, (uint32_t)strlen(utf8), 1, color);
    }
}

#define XL_HINDI_MAX_LINES 32U

uint32_t xl_hindi_text_draw_wrapped(int32_t x, int32_t y, uint32_t width, uint32_t max_lines, const char *utf8,
                                    uint16_t color)
{
    uint32_t start[XL_HINDI_MAX_LINES], len[XL_HINDI_MAX_LINES];
    uint32_t lines = 0, pos = 0, total, first, i;

    if ((hdr == NULL) || (utf8 == NULL) || (max_lines == 0U))
    {
        return 0;
    }
    total = (uint32_t)strlen(utf8);
    /* Greedy line breaking at spaces; a word wider than the box is cut between syllables. */
    while ((pos < total) && (lines < XL_HINDI_MAX_LINES))
    {
        uint32_t end = pos, best = pos, w;

        while (utf8[pos] == ' ')
        {
            pos++;
        }
        if (pos >= total)
        {
            break;
        }
        end = pos;
        for (;;)
        {
            uint32_t next = end;
            while ((next < total) && (utf8[next] == ' '))
            {
                next++;
            }
            while ((next < total) && (utf8[next] != ' '))
            {
                next++;
            }
            w = run(0, 0, utf8 + pos, next - pos, 0, 0);
            if ((w > width) && (best > pos))
            {
                break;
            }
            best = next;
            if ((w > width) || (next >= total))
            {
                break;
            }
            end = next;
        }
        if (run(0, 0, utf8 + pos, best - pos, 0, 0) > width)
        {
            /* one word wider than the box: take whole syllables while they fit */
            uint32_t cut = pos;
            for (;;)
            {
                const uint32_t l = xl_hindi_cluster_len(utf8 + cut);
                if ((l == 0U) || (run(0, 0, utf8 + pos, (cut + l) - pos, 0, 0) > width))
                {
                    break;
                }
                cut += l;
            }
            best = (cut > pos) ? cut : (pos + xl_hindi_cluster_len(utf8 + pos));
        }
        start[lines] = pos;
        len[lines] = best - pos;
        lines++;
        pos = best;
    }
    first = (lines > max_lines) ? (lines - max_lines) : 0U;
    for (i = first; i < lines; ++i)
    {
        const int32_t baseline = y + (int32_t)hdr->ascent + (int32_t)((i - first) * hdr->line_height);
        (void)run(x, baseline, utf8 + start[i], len[i], 1, color);
    }
    return lines - first;
}
