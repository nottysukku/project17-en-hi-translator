/**
 * @file     xl_speech.c
 * @brief    Hindi speech on the translator's speaker; implements xl_speech.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xl_speech.h"

#include <string.h>

#include "tts_front.h"
#include "tts_numbers.h"
#include "xl_audio_out.h"
#include "xl_resample.h"
#include "xl_voice_net.h"

#ifndef XL_SPEECH_HOST_TEST
#include "logger.h"
#include "tx_api.h"

static uint32_t now_ms(void)
{
    return (uint32_t)(((uint64_t)tx_time_get() * 1000U) / TX_TIMER_TICKS_PER_SECOND);
}

static void sleep_ms(uint32_t ms)
{
    (void)tx_thread_sleep((ULONG)(((ms * TX_TIMER_TICKS_PER_SECOND) + 999U) / 1000U));
}
#else
uint32_t xl_speech_host_now_ms(void);
void xl_speech_host_sleep_ms(uint32_t ms);
#define now_ms   xl_speech_host_now_ms
#define sleep_ms xl_speech_host_sleep_ms
static void log_none(const char *format, ...)
{
    (void)format;
}
#define LOG_INFO log_none
#endif

/** @brief Output level before the speaker ceiling (MMS-TTS peaks are below 1.0). */
#define XL_SPEECH_GAIN 0.9f

/** @brief Vocoder runs made before the first audio is queued (each is 1.024 s). */
#define XL_SPEECH_LEAD_RUNS 2U

/** @brief 16 kHz samples per latent frame, and 48 kHz samples per 16 kHz sample. */
#define XL_SPEECH_HOP 256U
#define XL_SPEECH_UP  3U

/** @brief 48 kHz samples of the longest sentence, with the resampler's tail. */
#define XL_SPEECH_PCM (((XL_SPEECH_MAX_FRAMES * XL_SPEECH_HOP) + XL_RESAMPLE_TAPS) * XL_SPEECH_UP)

/** @brief The sentence with its numbers spelled out (numbers grow: 25 -> पच्चीस). */
#define XL_SPEECH_TEXT 2048U

typedef struct
{
    tts_front_t front;
    char text[XL_SPEECH_TEXT];
    uint16_t ids[TTS_MAX_IDS];
    xl_resample_t resampler;
    float *acoustic; /* [XL_SPEECH_MAX_FRAMES][192]: the flow's output, frame-major */
    int16_t *pcm;    /* XL_SPEECH_PCM samples at 48 kHz */
    int ready;
} speech_t;

static speech_t sp;

static size_t align32(size_t n)
{
    return (n + 31U) & ~(size_t)31U;
}

size_t xl_speech_workspace_size(void)
{
    return align32(tts_front_workspace_size()) +
           align32((size_t)XL_SPEECH_MAX_FRAMES * XL_VOICE_CHANNELS * sizeof(float)) +
           align32((size_t)XL_SPEECH_PCM * sizeof(int16_t));
}

int xl_speech_init(const void *front_blob, void *workspace)
{
    uint8_t *w = (uint8_t *)workspace;
    int status;

    sp.ready = 0;
    if ((front_blob == NULL) || (workspace == NULL))
    {
        return -1;
    }
    if (tts_front_open(&sp.front, front_blob, w) != 0)
    {
        return -2; /* the front-end weights are not flashed */
    }
    w += align32(tts_front_workspace_size());
    sp.acoustic = (float *)(void *)w;
    w += align32((size_t)XL_SPEECH_MAX_FRAMES * XL_VOICE_CHANNELS * sizeof(float));
    sp.pcm = (int16_t *)(void *)w;
    status = xl_voice_net_init();
    if (status != 0)
    {
        return -10 + status; /* the voice networks are not built in, or failed */
    }
    sp.ready = 1;
    return 0;
}

int xl_speech_ready(void)
{
    return sp.ready;
}

static int cancelled(xl_speech_cancel_t cancel)
{
    return (cancel != NULL) && (cancel() != 0);
}

/** @brief Queue as much of pcm[*fed, made) as the speaker ring takes. */
static void feed(uint32_t *fed, uint32_t made)
{
    if (*fed < made)
    {
        *fed += xl_audio_out_write(sp.pcm + *fed, made - *fed);
    }
}

/** @brief The flow over frames [0, frames) in 512-frame windows with 32 frames of context. */
static int run_flow(uint32_t frames, xl_speech_cancel_t cancel)
{
    uint32_t done = 0, f, c, i;

    while (done < frames)
    {
        const uint32_t start = (done == 0U) ? 0U : (done - XL_VOICE_FLOW_CONTEXT);
        const uint32_t end = start + XL_VOICE_FLOW_FRAMES;
        const uint32_t trusted = (end >= frames) ? frames : (end - XL_VOICE_FLOW_CONTEXT);
        const xl_voice_tensor_t in = xl_voice_flow_latents();
        float *mask = xl_voice_flow_mask();
        xl_voice_tensor_t out;

        tts_front_latents(&sp.front, start, XL_VOICE_FLOW_FRAMES, in.base, in.c_stride, in.t_stride);
        for (i = 0; i < XL_VOICE_FLOW_FRAMES; ++i)
        {
            mask[i] = ((start + i) < frames) ? 1.0f : 0.0f;
        }
        if (xl_voice_flow_run(&out) != 0)
        {
            return -1;
        }
        for (f = done; f < trusted; ++f)
        {
            const float *src = out.base + ((size_t)(f - start) * out.t_stride);
            float *dst = sp.acoustic + ((size_t)f * XL_VOICE_CHANNELS);
            for (c = 0; c < XL_VOICE_CHANNELS; ++c)
            {
                dst[c] = src[(size_t)c * out.c_stride];
            }
        }
        done = trusted;
        if (cancelled(cancel))
        {
            return 1;
        }
    }
    return 0;
}

int xl_speech_say(const char *hindi, uint32_t seed, xl_speech_cancel_t cancel)
{
    static const float tail[XL_RESAMPLE_TAPS] = {0};
    uint32_t n, frames, start, made = 0, fed = 0, runs = 0, t0, t1, t2, i, c, waited;
    int status;

    if (!sp.ready || (hindi == NULL))
    {
        return -1;
    }
    (void)tts_spell_numbers(hindi, sp.text, sizeof(sp.text)); /* the voice knows few digits */
    n = tts_front_text_to_ids(&sp.front, sp.text, sp.ids, TTS_MAX_IDS);
    if (n == 0U)
    {
        return 0; /* nothing the voice can say */
    }
    t0 = now_ms();
    frames = tts_front_run(&sp.front, sp.ids, n, seed);
    if (frames == 0U)
    {
        return -2;
    }
    if (frames > XL_SPEECH_MAX_FRAMES)
    {
        frames = XL_SPEECH_MAX_FRAMES;
    }
    t1 = now_ms();
    if (cancelled(cancel))
    {
        return 1;
    }
    status = run_flow(frames, cancel);
    if (status != 0)
    {
        return (status > 0) ? 1 : -3;
    }
    t2 = now_ms();

    /* the vocoder, 64 frames per run with 15 frames of context each side, zero outside */
    xl_resample_reset(&sp.resampler);
    for (start = 0; start < frames; start += XL_VOICE_VOC_FRAMES)
    {
        const xl_voice_tensor_t in = xl_voice_vocoder_latents();
        const uint32_t valid = (((frames - start) < XL_VOICE_VOC_FRAMES) ? (frames - start) : XL_VOICE_VOC_FRAMES);
        const float *audio = NULL;

        for (i = 0; i < XL_VOICE_VOC_IN_FRAMES; ++i)
        {
            const int32_t f = (int32_t)(start + i) - (int32_t)XL_VOICE_VOC_CONTEXT;
            const int inside = (f >= 0) && (f < (int32_t)frames);
            const float *src = inside ? (sp.acoustic + ((size_t)f * XL_VOICE_CHANNELS)) : NULL;
            for (c = 0; c < XL_VOICE_CHANNELS; ++c)
            {
                in.base[((size_t)c * in.c_stride) + ((size_t)i * in.t_stride)] = inside ? src[c] : 0.0f;
            }
        }
        if (xl_voice_vocoder_run(&audio) != 0)
        {
            return -4;
        }
        xl_resample_run(&sp.resampler, audio, valid * XL_SPEECH_HOP, XL_SPEECH_GAIN, sp.pcm + made);
        made += valid * XL_SPEECH_HOP * XL_SPEECH_UP;
        ++runs;
        if ((start + XL_VOICE_VOC_FRAMES) >= frames)
        {
            /* the filter's last millisecond */
            xl_resample_run(&sp.resampler, tail, XL_RESAMPLE_TAPS, XL_SPEECH_GAIN, sp.pcm + made);
            made += XL_RESAMPLE_TAPS * XL_SPEECH_UP;
        }
        if ((runs >= XL_SPEECH_LEAD_RUNS) || ((start + XL_VOICE_VOC_FRAMES) >= frames))
        {
            feed(&fed, made);
        }
        if (cancelled(cancel))
        {
            return 1;
        }
    }
    LOG_INFO("Speech: %lu ids, %lu frames (%lu ms of audio): front %lu ms, flow %lu ms, vocoder %lu ms",
             (unsigned long)n, (unsigned long)frames, (unsigned long)((frames * 1000U) / 62U),
             (unsigned long)(t1 - t0), (unsigned long)(t2 - t1), (unsigned long)(now_ms() - t2));

    /* queue the rest as the ring drains, then wait until it has been played */
    waited = 0;
    while (fed < made)
    {
        feed(&fed, made);
        if (fed < made)
        {
            sleep_ms(20U);
        }
        if (cancelled(cancel))
        {
            return 1;
        }
    }
    while ((xl_audio_out_queued() > 0U) && (waited < 10000U))
    {
        sleep_ms(20U);
        waited += 20U;
        if (cancelled(cancel))
        {
            return 1;
        }
    }
    return 0;
}
