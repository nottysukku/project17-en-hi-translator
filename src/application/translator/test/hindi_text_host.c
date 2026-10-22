/**
 * @file     hindi_text_host.c
 * @brief    PC test of xl_hindi_text.c.
 *   hindi_text_host <font.bin> split <in.txt>       print syllable byte lengths per line
 *   hindi_text_host <font.bin> draw <in.txt> <out.pgm>  render the lines wrapped into 760 px
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "xl_hindi_text.h"

#define W 800
#define H 480
static unsigned char fb[H][W];

void mw_lcd_draw_alpha4(int32_t x, int32_t y, uint32_t width, uint32_t height, const uint8_t *bits, uint16_t color)
{
    uint32_t r, c, rb = (width + 1) / 2;
    (void)color;
    for (r = 0; r < height; ++r)
        for (c = 0; c < width; ++c)
        {
            int px = x + (int)c, py = y + (int)r;
            unsigned a = (c & 1) ? (bits[r * rb + c / 2] & 15) : (bits[r * rb + c / 2] >> 4);
            if (px >= 0 && px < W && py >= 0 && py < H && a * 17 > fb[py][px])
                fb[py][px] = (unsigned char)(a * 17);
        }
}

static void *load(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n;
    void *p;
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read\n"); exit(1); }
    fclose(f);
    return p;
}

int main(int argc, char **argv)
{
    char line[8192];
    FILE *in;
    int y = 10;
    if (argc < 4 || xl_hindi_text_open(load(argv[1])) != 0) { fprintf(stderr, "usage / bad font\n"); return 2; }
    in = fopen(argv[3], "rb");
    while (fgets(line, sizeof line, in))
    {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!strcmp(argv[2], "split"))
        {
            const char *p = line; uint32_t l; int first = 1;
            while ((l = xl_hindi_cluster_len(p)) != 0) { printf(first ? "%u" : " %u", l); first = 0; p += l; }
            printf("\n");
        }
        else
        {
            uint32_t lines = xl_hindi_text_draw_wrapped(20, y, 760, 4, line, 0xFFFF);
            y += (int)(lines * xl_hindi_text_line_height()) + 8;
        }
    }
    if (!strcmp(argv[2], "draw"))
    {
        FILE *o = fopen(argv[4], "wb");
        int r, c;
        fprintf(o, "P5 %d %d 255\n", W, H);
        for (r = 0; r < H; ++r) for (c = 0; c < W; ++c) fputc(255 - fb[r][c], o);
        fclose(o);
    }
    return 0;
}
