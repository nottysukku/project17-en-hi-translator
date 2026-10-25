/**
 * @file     feature_host.c
 * @brief    PC harness: the board's log-mel (mel.c) and normalization (norm.c) over a PCM file.
 * @details  Lets the training code prove that its features equal the board's
 *           (model-tools/asr_feature_check.py). Usage:
 *
 *               feature_host in.pcm mel.f32 windows.i8
 *
 *           in.pcm is mono 16 kHz int16 little-endian. The audio is fed in blocks of changing
 *           size, as the capture thread would. mel.f32 gets every log-mel frame (64 floats);
 *           windows.i8 gets every network input window norm.c writes (600 x 64 int8 codes).
 *
 *           Build (from firmware-n6/src/asr_engine):
 *               gcc -O2 -std=c11 -I. -Imel -Inorm -Itest -Imodels/cynipsnet_punct1 \
 *                   mel/mel.c mel/mel_fft.c mel/mel_tables.c norm/norm.c test/feature_host.c \
 *                   -lm -o feature_host
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>

#include "mel.h"
#include "norm.h"

static int8_t input[NORM_FIELD_VALUES];
static float field[NORM_FIELD_VALUES];
static norm_block_stats_t stats[NORM_STATS_BLOCKS];
static FILE *mel_out;
static FILE *window_out;
static unsigned long frames;
static unsigned long windows;

static void on_frame(const float *bands, void *context)
{
    (void)context;
    (void)fwrite(bands, sizeof(float), ASR_MODEL_MEL_BANDS, mel_out);
    frames++;
    if (norm_push_frame(bands) == NORM_WINDOW_READY)
    {
        (void)fwrite(input, 1, sizeof(input), window_out);
        windows++;
    }
}

int main(int argc, char **argv)
{
    const norm_buffers_t buffers = {input, field, stats};
    static int16_t pcm[1U << 16];
    FILE *in;
    size_t got;
    uint32_t seed = 12345U;

    if (argc != 4)
    {
        fprintf(stderr, "usage: %s in.pcm mel.f32 windows.i8\n", argv[0]);
        return 2;
    }
    in = fopen(argv[1], "rb");
    mel_out = fopen(argv[2], "wb");
    window_out = fopen(argv[3], "wb");
    if ((in == NULL) || (mel_out == NULL) || (window_out == NULL))
    {
        perror("open");
        return 1;
    }
    if ((mel_init() != MEL_OK) || (norm_init(&buffers) != NORM_OK))
    {
        fprintf(stderr, "init failed\n");
        return 1;
    }
    for (;;)
    {
        /* blocks of 1 to 700 samples, so frames straddle block edges */
        size_t want;
        seed = (seed * 1103515245U) + 12345U;
        want = 1U + ((seed >> 8) % 700U);
        got = fread(pcm, sizeof(int16_t), want, in);
        if (got == 0U)
        {
            break;
        }
        (void)mel_process(pcm, (uint32_t)got, on_frame, NULL);
    }
    fclose(in);
    fclose(mel_out);
    fclose(window_out);
    printf("%lu frames, %lu windows\n", frames, windows);
    return 0;
}
