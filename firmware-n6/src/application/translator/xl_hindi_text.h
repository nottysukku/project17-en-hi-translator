/**
 * @file     xl_hindi_text.h
 * @brief    Devanagari text on the LCD from a pre-shaped syllable font (hi_font_build.py).
 * @details  Devanagari letters join, reorder and stack within a syllable, which needs a shaping
 *           engine (HarfBuzz). Instead, every syllable that occurs in a large Hindi corpus was
 *           shaped and rasterised on the PC (Noto Sans Devanagari, 4-bit alpha). Here text is
 *           split into syllables with the same rules, each is looked up and blended into the
 *           framebuffer (mw_lcd_draw_alpha4). A syllable missing from the table is drawn
 *           letter by letter.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_HINDI_TEXT_H
#define APPLICATION_TRANSLATOR_XL_HINDI_TEXT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Use a syllable font blob (memory-mapped).
 * @return 0 on success, negative if the blob is not a HIFT font.
 */
int xl_hindi_text_open(const void *blob);

/** @brief Line advance in pixels (0 if no font is open). */
uint32_t xl_hindi_text_line_height(void);

/** @brief Height above the baseline in pixels. */
uint32_t xl_hindi_text_ascent(void);

/** @brief Width in pixels of UTF-8 text drawn on one line. */
uint32_t xl_hindi_text_width(const char *utf8);

/** @brief Draw UTF-8 text on one line; @p baseline is the y of the baseline. */
void xl_hindi_text_draw(int32_t x, int32_t baseline, const char *utf8, uint16_t color);

/**
 * @brief Draw a paragraph wrapped at spaces into a box.
 * @param[in] x Left edge of the box.
 * @param[in] y Top edge of the box.
 * @param[in] width Box width in pixels.
 * @param[in] max_lines Lines available; text that does not fit is dropped from the start, so
 *            the end of a long translation stays visible.
 * @return Lines drawn.
 */
uint32_t xl_hindi_text_draw_wrapped(int32_t x, int32_t y, uint32_t width, uint32_t max_lines, const char *utf8,
                                    uint16_t color);

/**
 * @brief Length in bytes of the syllable starting at @p s (exposed for testing).
 * @return 0 at the end of the string.
 */
uint32_t xl_hindi_cluster_len(const char *s);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_HINDI_TEXT_H */
