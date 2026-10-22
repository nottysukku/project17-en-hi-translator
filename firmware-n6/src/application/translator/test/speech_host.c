/**
 * @file     speech_host.c
 * @brief    PC test of xl_speech.c with stand-in NPU networks and speaker ring.
 *   speech_host <tts_front.bin>
 * The flow is the identity and the "vocoder" turns latent frame f into 256 samples of
 * 0.01 * channel 0 of frame f, so the speech must equal the front end's latents, frame after
 * frame, through the resampler: this checks every window, context and chunk boundary.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tts_front.h"
#include "xl_audio_out.h"
#include "xl_resample.h"
#include "xl_speech.h"
#include "xl_voice_net.h"

/* ---- time and the speaker ring: 4 s at 48 kHz, 960 samples played per 20 ms ---- */
static uint32_t clock_ms;
static int16_t *played;
static uint32_t n_played, n_queued;
#define RING (4U * 48000U)

uint32_t xl_speech_host_now_ms(void) { return clock_ms; }
void xl_speech_host_sleep_ms(uint32_t ms)
{
    uint32_t drain = 48U * ms;
    clock_ms += ms;
    n_queued = (n_queued > drain) ? n_queued - drain : 0U;
}
uint32_t xl_audio_out_write(const int16_t *samples, uint32_t count)
{
    uint32_t k = RING - n_queued;
    if (k > count) k = count;
    memcpy(played + n_played, samples, k * sizeof(int16_t));
    n_played += k;
    n_queued += k;
    return k;
}
uint32_t xl_audio_out_queued(void) { return n_queued; }

/* ---- stand-in networks, channel-last like the generated code ---- */
static float flow_in[XL_VOICE_FLOW_FRAMES * XL_VOICE_CHANNELS], flow_mask[XL_VOICE_FLOW_FRAMES];
static float flow_out[XL_VOICE_FLOW_FRAMES * XL_VOICE_CHANNELS];
static float voc_in[XL_VOICE_VOC_IN_FRAMES * XL_VOICE_CHANNELS], voc_out[XL_VOICE_VOC_FRAMES * 256U];
static int flow_runs, voc_runs, cancel_after = -1;

int xl_voice_net_init(void) { return 0; }
int xl_voice_net_ready(void) { return 1; }
xl_voice_tensor_t xl_voice_flow_latents(void) { xl_voice_tensor_t t = {flow_in, 1U, XL_VOICE_CHANNELS}; return t; }
float *xl_voice_flow_mask(void) { return flow_mask; }
int xl_voice_flow_run(xl_voice_tensor_t *out)
{
    uint32_t i;
    for (i = 0; i < XL_VOICE_FLOW_FRAMES * XL_VOICE_CHANNELS; ++i)
        flow_out[i] = flow_in[i] * flow_mask[i / XL_VOICE_CHANNELS];
    out->base = flow_out; out->c_stride = 1U; out->t_stride = XL_VOICE_CHANNELS;
    flow_runs++;
    return 0;
}
xl_voice_tensor_t xl_voice_vocoder_latents(void) { xl_voice_tensor_t t = {voc_in, 1U, XL_VOICE_CHANNELS}; return t; }
int xl_voice_vocoder_run(const float **audio)
{
    uint32_t j, s;
    for (j = 0; j < XL_VOICE_VOC_FRAMES; ++j)
        for (s = 0; s < 256U; ++s)
            voc_out[j * 256U + s] = 0.01f * voc_in[(j + XL_VOICE_VOC_CONTEXT) * XL_VOICE_CHANNELS];
    *audio = voc_out;
    voc_runs++;
    return 0;
}
static int cancel_check(void) { return (cancel_after >= 0) && (voc_runs >= cancel_after); }

static void *load(const char *path)
{
    FILE *f = fopen(path, "rb"); long n; void *p;
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    p = malloc((size_t)n);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) exit(1);
    fclose(f);
    return p;
}

static int check(const void *blob, const char *text, uint32_t seed)
{
    static uint16_t ids[TTS_MAX_IDS];
    tts_front_t ref;
    void *ws = calloc(1, tts_front_workspace_size());
    uint32_t n, frames, f, k, i, mismatch = 0;
    float *z, *wave;
    int16_t *expect;
    xl_resample_t rs;
    static const float zeros[XL_RESAMPLE_TAPS];

    /* what the speech must be: the front end's latents, channel 0, 256 samples per frame */
    tts_front_open(&ref, blob, ws);
    n = tts_front_text_to_ids(&ref, text, ids, TTS_MAX_IDS);
    frames = tts_front_run(&ref, ids, n, seed);
    z = malloc((size_t)frames * TTS_CHANNELS * sizeof(float));
    tts_front_latents(&ref, 0, frames, z, frames, 1);
    wave = malloc((size_t)frames * 256U * sizeof(float));
    for (f = 0; f < frames; ++f)
        for (k = 0; k < 256U; ++k)
            wave[f * 256U + k] = 0.01f * z[f];
    expect = malloc(((size_t)frames * 256U + XL_RESAMPLE_TAPS) * 3U * sizeof(int16_t));
    xl_resample_reset(&rs);
    for (f = 0; f < frames; f += 64U)
    {
        const uint32_t v = (frames - f < 64U) ? frames - f : 64U;
        xl_resample_run(&rs, wave + f * 256U, v * 256U, 0.9f, expect + f * 256U * 3U);
    }
    xl_resample_run(&rs, zeros, XL_RESAMPLE_TAPS, 0.9f, expect + frames * 256U * 3U);

    n_played = n_queued = 0; flow_runs = voc_runs = 0;
    if (xl_speech_say(text, seed, cancel_check) != 0) { printf("FAIL: say() failed\n"); return 1; }
    for (i = 0; i < n_played; ++i)
        mismatch += (played[i] != expect[i]);
    printf("%u frames: %d flow runs, %d vocoder runs, %u samples at 48 kHz (expected %u), %u differ, ring %s\n",
           frames, flow_runs, voc_runs, n_played, (frames * 256U + XL_RESAMPLE_TAPS) * 3U, mismatch,
           n_queued ? "NOT drained" : "drained");
    free(ws); free(z); free(wave); free(expect);
    return (mismatch != 0) || (n_played != (frames * 256U + XL_RESAMPLE_TAPS) * 3U) || (n_queued != 0);
}

int main(int argc, char **argv)
{
    const void *blob;
    void *ws;
    char long_text[4096] = "";
    int fail = 0, i, r;

    if (argc < 2) { fprintf(stderr, "usage: speech_host <tts_front.bin>\n"); return 2; }
    blob = load(argv[1]);
    played = malloc(XL_SPEECH_MAX_FRAMES * 256U * 3U * sizeof(int16_t) + 4096);
    ws = calloc(1, xl_speech_workspace_size());
    if (xl_speech_init(blob, ws) != 0) { printf("FAIL: init\n"); return 1; }
    fail |= check(blob, "नमस्ते, आप कैसे हैं?", 7);
    /* longer than one 512-frame flow window */
    for (i = 0; i < 6; ++i)
        strcat(long_text, "मेरा भाई बैंगलोर में सॉफ्टवेयर इंजीनियर के रूप में काम करता है। ");
    long_text[strlen(long_text) / 2] = 0; /* keep it under the 255-character limit */
    while ((unsigned char)long_text[strlen(long_text) - 1] >= 0x80 && ((unsigned char)long_text[strlen(long_text) - 1] & 0xC0) == 0x80)
        long_text[strlen(long_text) - 1] = 0;
    long_text[strlen(long_text) - 1] = 0;
    fail |= check(blob, long_text, 11);
    /* cancelled after the first vocoder run */
    cancel_after = 1; n_played = n_queued = 0; voc_runs = 0;
    r = xl_speech_say("मुझे समझ नहीं आ रहा है कि आप क्या कह रहे हैं।", 3, cancel_check);
    printf("cancel after one vocoder run: say() = %d, %d vocoder runs, %u samples queued\n", r, voc_runs, n_played);
    fail |= (r != 1) || (voc_runs != 1) || (n_played != 0);
    printf(fail ? "SPEECH TEST FAILED\n" : "all speech tests passed\n");
    return fail;
}
