/**
 * @file test_beam.c
 * @brief Comprehensive test suite for Cortex-M55 CTC Prefix Beam Search decoder.
 */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "net_ctc_beam.h"

int main(void)
{
    printf("Running net_ctc_beam unit test suite...\n");

    net_beam_state_t state;
    net_ctc_beam_init(&state, 8, 3.5f);

    const char *tok = ASR_MODEL_NET_TOKENS;
    int idx_s = strchr(tok, 'S') - tok;
    int idx_t = strchr(tok, 'T') - tok;
    int idx_o = strchr(tok, 'O') - tok;
    int idx_p = strchr(tok, 'P') - tok;
    int idx_a = strchr(tok, 'A') - tok;
    int idx_r = strchr(tok, 'R') - tok;
    int idx_e = strchr(tok, 'E') - tok;
    int idx_v = strchr(tok, 'V') - tok;
    int idx_l = strchr(tok, 'L') - tok;
    int idx_u = strchr(tok, 'U') - tok;
    int idx_m = strchr(tok, 'M') - tok;
    int idx_h = strchr(tok, 'H') - tok;
    int idx_w = strchr(tok, 'W') - tok;
    int idx_d = strchr(tok, 'D') - tok;
    int idx_sp = ASR_MODEL_NET_SPACE_INDEX;

    /* Test 1: Empty logits */
    char text[128];
    uint32_t len = net_ctc_beam_decode(&state, NULL, 0, 0, text, sizeof(text));
    assert(len == 0);
    assert(text[0] == '\0');
    printf("Test 1 (empty logits) PASSED\n");

    /* Test 2: Synthetic clear logits for 'STOP' */
    int8_t logits[ASR_NETWORK_OUTPUT_STEPS * ASR_NETWORK_OUTPUT_CLASSES];
    memset(logits, -50, sizeof(logits));

    logits[0 * ASR_NETWORK_OUTPUT_CLASSES + 0] = 50; /* blank */
    logits[1 * ASR_NETWORK_OUTPUT_CLASSES + idx_s] = 50;
    logits[2 * ASR_NETWORK_OUTPUT_CLASSES + idx_t] = 50;
    logits[3 * ASR_NETWORK_OUTPUT_CLASSES + idx_o] = 50;
    logits[4 * ASR_NETWORK_OUTPUT_CLASSES + idx_p] = 50;
    logits[5 * ASR_NETWORK_OUTPUT_CLASSES + 0] = 50; /* blank */

    net_ctc_beam_reset(&state);
    len = net_ctc_beam_decode(&state, logits, 0, 6, text, sizeof(text));
    printf("Test 2 Decoded: '%s' (len=%lu)\n", text, (unsigned long)len);
    assert(strcmp(text, "STOP") == 0);
    printf("Test 2 (STOP sequence) PASSED\n");

    /* Test 3: Hotword recovery under noisy acoustic logits ('START' vs 'STATE') */
    memset(logits, -50, sizeof(logits));
    logits[0 * ASR_NETWORK_OUTPUT_CLASSES + idx_s] = 40;
    logits[1 * ASR_NETWORK_OUTPUT_CLASSES + idx_t] = 40;
    logits[2 * ASR_NETWORK_OUTPUT_CLASSES + idx_a] = 40;
    logits[3 * ASR_NETWORK_OUTPUT_CLASSES + idx_t] = 35; /* T favored over R */
    logits[3 * ASR_NETWORK_OUTPUT_CLASSES + idx_r] = 30;
    logits[4 * ASR_NETWORK_OUTPUT_CLASSES + idx_e] = 35; /* E favored over T -> 'STATE' */
    logits[4 * ASR_NETWORK_OUTPUT_CLASSES + idx_t] = 30;

    net_ctc_beam_clear_hotwords();
    net_ctc_beam_add_hotword("START", 5.0f);

    net_ctc_beam_reset(&state);
    len = net_ctc_beam_decode(&state, logits, 0, 5, text, sizeof(text));
    printf("Test 3 Decoded with hotword START: '%s' (len=%lu)\n", text, (unsigned long)len);
    assert(strcmp(text, "START") == 0);
    printf("Test 3 (Hotword START recovery over STATE) PASSED\n");

    /* Test 4: Multi-word phrase hotword recovery ('VOLUME UP') */
    /* V, O, L, U, M, E, ' ', U, P */
    memset(logits, -50, sizeof(logits));
    logits[0 * ASR_NETWORK_OUTPUT_CLASSES + idx_v] = 45;
    logits[1 * ASR_NETWORK_OUTPUT_CLASSES + idx_o] = 45;
    logits[2 * ASR_NETWORK_OUTPUT_CLASSES + idx_l] = 45;
    logits[3 * ASR_NETWORK_OUTPUT_CLASSES + idx_u] = 45;
    logits[4 * ASR_NETWORK_OUTPUT_CLASSES + idx_m] = 45;
    logits[5 * ASR_NETWORK_OUTPUT_CLASSES + idx_e] = 45;
    logits[6 * ASR_NETWORK_OUTPUT_CLASSES + idx_sp] = 45;
    logits[7 * ASR_NETWORK_OUTPUT_CLASSES + idx_u] = 45;
    logits[8 * ASR_NETWORK_OUTPUT_CLASSES + idx_p] = 45;
    logits[9 * ASR_NETWORK_OUTPUT_CLASSES + idx_sp] = 45;

    net_ctc_beam_clear_hotwords();
    net_ctc_beam_add_hotword("VOLUME UP", 4.0f);

    net_ctc_beam_reset(&state);
    len = net_ctc_beam_decode(&state, logits, 0, 10, text, sizeof(text));
    printf("Test 4 Multi-word phrase decoded: '%s' (len=%lu)\n", text, (unsigned long)len);
    assert(strcmp(text, "VOLUME UP ") == 0);
    printf("Test 4 (Multi-word phrase 'VOLUME UP') PASSED\n");

    /* Test 5: Pause-Space insertion between words separated by >= 15 blank steps */
    memset(logits, -50, sizeof(logits));
    /* 'START' at steps 0..4 */
    logits[0 * ASR_NETWORK_OUTPUT_CLASSES + idx_s] = 45;
    logits[1 * ASR_NETWORK_OUTPUT_CLASSES + idx_t] = 45;
    logits[2 * ASR_NETWORK_OUTPUT_CLASSES + idx_a] = 45;
    logits[3 * ASR_NETWORK_OUTPUT_CLASSES + idx_r] = 45;
    logits[4 * ASR_NETWORK_OUTPUT_CLASSES + idx_t] = 45;
    /* 16 steps of blank (steps 5..20) */
    int s;
    for (s = 5; s <= 20; ++s)
    {
        logits[s * ASR_NETWORK_OUTPUT_CLASSES + 0] = 45; /* blank */
    }
    /* 'STOP' at steps 21..24 */
    logits[21 * ASR_NETWORK_OUTPUT_CLASSES + idx_s] = 45;
    logits[22 * ASR_NETWORK_OUTPUT_CLASSES + idx_t] = 45;
    logits[23 * ASR_NETWORK_OUTPUT_CLASSES + idx_o] = 45;
    logits[24 * ASR_NETWORK_OUTPUT_CLASSES + idx_p] = 45;

    net_ctc_beam_clear_hotwords();
    net_ctc_beam_reset(&state);
    len = net_ctc_beam_decode(&state, logits, 0, 25, text, sizeof(text));
    printf("Test 5 Pause-space decoded: '%s' (len=%lu)\n", text, (unsigned long)len);
    assert(strcmp(text, "START STOP") == 0);
    printf("Test 5 (Pause-space insertion 'START STOP') PASSED\n");

    /* Test 6: Streaming multi-window decoding (verifying delta output and no repetition) */
    /* Window 1: steps 0..4 produce 'HELLO ' */
    memset(logits, -50, sizeof(logits));
    logits[0 * ASR_NETWORK_OUTPUT_CLASSES + idx_h] = 45;
    logits[1 * ASR_NETWORK_OUTPUT_CLASSES + idx_e] = 45;
    logits[2 * ASR_NETWORK_OUTPUT_CLASSES + idx_l] = 45;
    logits[3 * ASR_NETWORK_OUTPUT_CLASSES + 0]     = 45; /* blank to separate repeated L */
    logits[4 * ASR_NETWORK_OUTPUT_CLASSES + idx_l] = 45;
    logits[5 * ASR_NETWORK_OUTPUT_CLASSES + idx_o] = 45;
    logits[6 * ASR_NETWORK_OUTPUT_CLASSES + idx_sp] = 45;

    net_ctc_beam_reset(&state);
    char win1_text[64];
    uint32_t len1 = net_ctc_beam_decode(&state, logits, 0, 7, win1_text, sizeof(win1_text));
    printf("Test 6 Window 1 output: '%s' (len=%lu)\n", win1_text, (unsigned long)len1);
    assert(strcmp(win1_text, "HELLO ") == 0);

    /* Window 2: steps 7..11 produce 'WORLD' */
    logits[7 * ASR_NETWORK_OUTPUT_CLASSES + idx_w] = 45;
    logits[8 * ASR_NETWORK_OUTPUT_CLASSES + idx_o] = 45;
    logits[9 * ASR_NETWORK_OUTPUT_CLASSES + idx_r] = 45;
    logits[10 * ASR_NETWORK_OUTPUT_CLASSES + idx_l] = 45;
    logits[11 * ASR_NETWORK_OUTPUT_CLASSES + idx_d] = 45;

    char win2_text[64];
    uint32_t len2 = net_ctc_beam_decode(&state, logits, 7, 5, win2_text, sizeof(win2_text));
    printf("Test 6 Window 2 output: '%s' (len=%lu)\n", win2_text, (unsigned long)len2);
    /* Window 2 must return ONLY the delta 'WORLD', NOT 'HELLO WORLD'! */
    assert(strcmp(win2_text, "WORLD") == 0);
    printf("Test 6 (Streaming window delta output) PASSED\n");

    /* Test 7: Long continuous streaming (simulate 50 consecutive windows) */
    net_ctc_beam_reset(&state);
    uint32_t total_chars = 0;
    int w;
    for (w = 0; w < 50; ++w)
    {
        /* Alternate emitting "YES " and "NO " */
        memset(logits, -50, sizeof(logits));
        if (w % 2 == 0)
        {
            int idx_y = strchr(tok, 'Y') - tok;
            logits[0 * ASR_NETWORK_OUTPUT_CLASSES + idx_y] = 45;
            logits[1 * ASR_NETWORK_OUTPUT_CLASSES + idx_e] = 45;
            logits[2 * ASR_NETWORK_OUTPUT_CLASSES + idx_s] = 45;
            logits[3 * ASR_NETWORK_OUTPUT_CLASSES + idx_sp] = 45;
        }
        else
        {
            int idx_n = strchr(tok, 'N') - tok;
            logits[0 * ASR_NETWORK_OUTPUT_CLASSES + idx_n] = 45;
            logits[1 * ASR_NETWORK_OUTPUT_CLASSES + idx_o] = 45;
            logits[2 * ASR_NETWORK_OUTPUT_CLASSES + idx_sp] = 45;
        }
        char w_out[32];
        uint32_t w_len = net_ctc_beam_decode(&state, logits, 0, 4, w_out, sizeof(w_out));
        total_chars += w_len;
        assert(state.entries[0].length < 64); /* Must stay tightly bounded! */
    }
    printf("Test 7 Continuous 50 windows: total_chars=%lu, active_tail_len=%u\n",
           (unsigned long)total_chars, state.entries[0].length);
    assert(total_chars > 100);
    assert(state.entries[0].length < 64);
    printf("Test 7 (Long continuous stream stability) PASSED\n");

    printf("\nALL NET_CTC_BEAM TESTS PASSED CLEANLY WITH ZERO ERRORS!\n");
    return 0;
}
