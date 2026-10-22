/**
 * @file     it2_host.c
 * @brief    PC test driver for the it2 engine: translate one English sentence per input line.
 *
 *   it2_host <model.bin> <in.txt> [--ids]
 * Prints "<hindi>\t<src tokens>\t<tgt tokens>\t<encode ms>\t<decode ms>" per line, or the source
 * token ids with --ids (to check the tokeniser against the Python reference).
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "it2.h"

uint32_t it2_now_ms(void)
{
    return (uint32_t)((clock() * 1000.0) / CLOCKS_PER_SEC);
}

int main(int argc, char **argv)
{
    FILE *f;
    long size;
    void *blob, *ws;
    it2_model_t m;
    char line[2048], out[4096];
    const int ids_only = (argc > 3) && (strcmp(argv[3], "--ids") == 0);

    if (argc < 3)
    {
        fprintf(stderr, "usage: it2_host <model.bin> <in.txt> [--ids]\n");
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (f == NULL)
    {
        perror(argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);
    blob = malloc((size_t)size);
    if ((blob == NULL) || (fread(blob, 1, (size_t)size, f) != (size_t)size))
    {
        fprintf(stderr, "read failed\n");
        return 1;
    }
    fclose(f);
    ws = calloc(1, it2_workspace_size(blob));
    if ((ws == NULL) || (it2_open(&m, blob, ws) != 0))
    {
        fprintf(stderr, "it2_open failed\n");
        return 1;
    }
    fprintf(stderr, "model: %u enc + %u dec layers, %u target tokens, workspace %.1f MB\n", m.hdr->n_enc,
            m.hdr->n_dec, m.hdr->tgt_vocab, it2_workspace_size(blob) / 1e6);
    f = fopen(argv[2], "rb");
    if (f == NULL)
    {
        perror(argv[2]);
        return 1;
    }
    while (fgets(line, sizeof(line), f) != NULL)
    {
        size_t n = strlen(line);
        while ((n > 0U) && ((line[n - 1U] == '\n') || (line[n - 1U] == '\r')))
        {
            line[--n] = '\0';
        }
        if (ids_only)
        {
            uint32_t ids[IT2_MAX_SRC], k, c = it2_encode_text(&m, line, ids, IT2_MAX_SRC);
            for (k = 0; k < c; ++k)
            {
                printf("%s%u", k ? " " : "", ids[k]);
            }
            printf("\n");
        }
        else
        {
            it2_stats_t st;
            it2_translate(&m, line, out, sizeof(out), &st);
            printf("%s\t%u\t%u\t%u\t%u\n", out, st.src_tokens, st.tgt_tokens, st.encode_ms, st.decode_ms);
        }
        fflush(stdout);
    }
    fclose(f);
    return 0;
}
