/**
 * @file     tts_front_host.c
 * @brief    PC test driver for tts_front.c: run the front end on each input line.
 *
 *   tts_front_host <tts_front.bin> <in.txt> <seed> <out_dir>
 * For line i writes <out_dir>/front_<i>.bin: u32 ids, u32 frames, the ids (u16), the durations
 * (u16), logw (f32 per id), the prior stats (f32 [ids][384]: means, then log-variances) and the
 * latents (f32 [192][frames]). Prints "<ids>\t<frames>\t<ms>" per line.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tts_front.h"

static void *load(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n;
    void *p;

    if (f == NULL)
    {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    p = malloc((size_t)n);
    if ((p == NULL) || (fread(p, 1, (size_t)n, f) != (size_t)n))
    {
        fprintf(stderr, "read failed\n");
        exit(1);
    }
    fclose(f);
    return p;
}

int main(int argc, char **argv)
{
    static uint16_t ids[TTS_MAX_IDS];
    tts_front_t fe;
    char line[4096], path[1024];
    FILE *in;
    void *ws;
    unsigned seed;
    int index = 0;

    if (argc < 5)
    {
        fprintf(stderr, "usage: tts_front_host <tts_front.bin> <in.txt> <seed> <out_dir>\n");
        return 2;
    }
    seed = (unsigned)strtoul(argv[3], NULL, 0);
    ws = calloc(1, tts_front_workspace_size());
    if ((ws == NULL) || (tts_front_open(&fe, load(argv[1]), ws) != 0))
    {
        fprintf(stderr, "tts_front_open failed\n");
        return 1;
    }
    fprintf(stderr, "workspace %.2f MB\n", tts_front_workspace_size() / 1e6);
    in = fopen(argv[2], "rb");
    if (in == NULL)
    {
        perror(argv[2]);
        return 1;
    }
    while (fgets(line, sizeof(line), in) != NULL)
    {
        size_t len = strlen(line);
        uint32_t n, frames = 0;
        clock_t t0;
        FILE *o;
        float *z;

        while ((len > 0U) && ((line[len - 1U] == '\n') || (line[len - 1U] == '\r')))
        {
            line[--len] = '\0';
        }
        n = tts_front_text_to_ids(&fe, line, ids, TTS_MAX_IDS);
        t0 = clock();
        if (n > 0U)
        {
            frames = tts_front_run(&fe, ids, n, seed + (unsigned)index);
        }
        printf("%u\t%u\t%.0f\n", n, frames, (double)(clock() - t0) * 1000.0 / CLOCKS_PER_SEC);
        snprintf(path, sizeof(path), "%s/front_%02d.bin", argv[4], index);
        o = fopen(path, "wb");
        if (o == NULL)
        {
            perror(path);
            return 1;
        }
        fwrite(&n, 4, 1, o);
        fwrite(&frames, 4, 1, o);
        fwrite(ids, 2, n, o);
        if (frames > 0U)
        {
            fwrite(tts_front_durations(&fe), 2, n, o);
            fwrite(tts_front_logw(&fe), 4, n, o);
            fwrite(tts_front_prior_mean(&fe), 4, (size_t)n * 384U, o);
            z = malloc((size_t)frames * TTS_CHANNELS * sizeof(float));
            tts_front_latents(&fe, 0, frames, z, frames, 1);
            fwrite(z, 4, (size_t)frames * TTS_CHANNELS, o);
            free(z);
        }
        fclose(o);
        index++;
    }
    fclose(in);
    return 0;
}
