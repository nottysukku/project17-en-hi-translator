/**
 * @file     xl_translator.c
 * @brief    The translator thread; implements xl_translator.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xl_translator.h"

#include <string.h>

#include "app_threads.h"
#include "asr_engine.h"
#include "audio_port.h"
#include "it2.h"
#include "logger.h"
#include "npu_cache.h"
#include "stm32n6xx_hal.h"
#include "text_case.h"
#include "tx_api.h"
#include "xl_models.h"
#include "xl_router.h"
#include "xl_sentence.h"
#include "xl_speech.h"

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief Sentences waiting for the translator. */
#define XL_QUEUE_LENGTH 4U

/** @brief Hindi output buffer for one sentence. */
#define XL_HINDI_SENTENCE 768U

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

static TX_THREAD thread;
static ULONG thread_stack[APP_TRANSLATOR_THREAD_STACK_BYTES / sizeof(ULONG)];
static TX_MUTEX lock;          /* queue, Hindi text, listen reasons */
static TX_SEMAPHORE wake;      /* a sentence was queued */

static char queue[XL_QUEUE_LENGTH][XL_SENTENCE_CAPACITY];
static uint32_t queue_head, queue_count;

static char hindi[XL_TRANSLATOR_HINDI_CAPACITY];
static uint32_t hindi_revision;

static uint32_t listen_reasons;
static volatile uint32_t cancel_epoch;
static volatile uint32_t job_epoch;
static volatile uint32_t activity;

static it2_model_t model;
static int model_ok;
static int speech_ok;
static uint32_t seed = 1U;

/* Working copies, owned by the thread. */
static char english[XL_SENTENCE_CAPACITY];
static char spoken[XL_HINDI_SENTENCE];

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief The translation engine's clock (it2.h), for its timing statistics. */
uint32_t it2_now_ms(void)
{
    return (uint32_t)(((uint64_t)tx_time_get() * 1000U) / TX_TIMER_TICKS_PER_SECOND);
}

/** @brief Polled by it2 once per token. */
static int translation_cancelled(void)
{
    return job_epoch != cancel_epoch;
}

/** @brief Polled by the speech between steps: cancel, or the speaker is going to the music. */
static int speech_cancelled(void)
{
    return (job_epoch != cancel_epoch) || (xl_router_target() != XL_SOURCE_N6);
}

/**
 * @brief The PSRAM work region holds the voice now. The CPU's D-cache is written back, so no dirty
 *        line of the translation workspace can land on the voice networks' activations; the NPU
 *        cache drops its lines of the region (none are dirty: see leave_voice_phase()).
 */
static void enter_voice_phase(void)
{
    SCB_CleanInvalidateDCache();
    npu_cache_clean_invalidate_range(XL_PSRAM_WORK_ADDR, XL_PSRAM_WORK_ADDR + XL_PSRAM_WORK_SIZE);
}

/**
 * @brief The translation gets the work region back. The NPU cache writes back and drops its lines
 *        of the region now, while the data is still the voice's: otherwise the recognizer's NPU
 *        traffic could evict them later, over the translation workspace.
 */
static void leave_voice_phase(void)
{
    npu_cache_clean_invalidate_range(XL_PSRAM_WORK_ADDR, XL_PSRAM_WORK_ADDR + XL_PSRAM_WORK_SIZE);
}

static int take_sentence(void)
{
    int got = 0;

    (void)tx_mutex_get(&lock, TX_WAIT_FOREVER);
    if (queue_count > 0U)
    {
        (void)strcpy(english, queue[queue_head]);
        queue_head = (queue_head + 1U) % XL_QUEUE_LENGTH;
        queue_count--;
        job_epoch = cancel_epoch;
        got = 1;
    }
    (void)tx_mutex_put(&lock);
    return got;
}

static void publish_hindi(const char *text)
{
    (void)tx_mutex_get(&lock, TX_WAIT_FOREVER);
    (void)strncpy(hindi, text, sizeof(hindi) - 1U);
    hindi[sizeof(hindi) - 1U] = '\0';
    hindi_revision++;
    (void)tx_mutex_put(&lock);
}

/** @brief Open the models once: check the translation model, start the voice. */
static void open_models(void)
{
    const void *blob = (const void *)XL_MODEL_TRANSLATION_ADDR;
    size_t mt_ws = 0U;

    /* erased flash reads 0xFF: look at the magic before trusting any size in the header */
    if (memcmp(blob, "IT2B", 4) == 0)
    {
        mt_ws = it2_workspace_size(blob);
        model_ok = (mt_ws <= XL_PSRAM_WORK_SIZE) && (it2_open(&model, blob, (void *)XL_PSRAM_WORK_ADDR) == 0);
    }
    if (model_ok)
    {
        LOG_INFO("Translator: model %lu enc + %lu dec layers, %lu Hindi tokens, workspace %lu KB",
                 (unsigned long)model.hdr->n_enc, (unsigned long)model.hdr->n_dec,
                 (unsigned long)model.hdr->tgt_vocab, (unsigned long)(mt_ws / 1024U));
    }
    else
    {
        LOG_ERROR("Translator: no translation model at 0x%08lx (flash it with flash.py n6 --models)",
                  (unsigned long)XL_MODEL_TRANSLATION_ADDR);
    }

    if (xl_speech_workspace_size() > XL_PSRAM_VOICE_WORK_SIZE)
    {
        LOG_ERROR("Translator: speech workspace too large for its PSRAM region");
        return;
    }
    asr_engine_npu_acquire();
    enter_voice_phase();
    {
        const int status = xl_speech_init((const void *)XL_MODEL_TTS_FRONT_ADDR, (void *)XL_PSRAM_WORK_ADDR);
        speech_ok = (status == 0);
        if (speech_ok)
        {
            LOG_INFO("Translator: Hindi voice ready");
        }
        else
        {
            LOG_WARNING("Translator: no Hindi voice (%d): text only", status);
        }
    }
    leave_voice_phase();
    asr_engine_npu_release();
}

static void translate_and_speak(void)
{
    it2_stats_t stats;
    int length;

    text_sentence_case(english, (uint32_t)strlen(english), 1);
    activity = 1U;
    /* the voice may have used the work region since: open the model again (cheap) */
    if (it2_open(&model, (const void *)XL_MODEL_TRANSLATION_ADDR, (void *)XL_PSRAM_WORK_ADDR) != 0)
    {
        activity = 0U;
        return;
    }
    it2_set_cancel(&model, translation_cancelled);
    length = it2_translate(&model, english, spoken, sizeof(spoken), &stats);
    if (length == IT2_CANCELLED)
    {
        LOG_INFO("Translator: translation cancelled");
        activity = 0U;
        return;
    }
    if (length <= 0)
    {
        activity = 0U;
        return;
    }
    LOG_INFO("Translator: \"%s\" -> \"%s\" (%lu + %lu tokens, encode %lu ms, decode %lu ms)", english, spoken,
             (unsigned long)stats.src_tokens, (unsigned long)stats.tgt_tokens, (unsigned long)stats.encode_ms,
             (unsigned long)stats.decode_ms);
    publish_hindi(spoken);

    if (!speech_ok || speech_cancelled())
    {
        activity = 0U;
        return;
    }
    activity = 2U;
    xl_translator_listen(XL_LISTEN_SPEECH, 1);
    asr_engine_npu_acquire();
    enter_voice_phase();
    if (xl_speech_init((const void *)XL_MODEL_TTS_FRONT_ADDR, (void *)XL_PSRAM_WORK_ADDR) == 0)
    {
        const int status = xl_speech_say(spoken, seed++, speech_cancelled);
        if (status < 0)
        {
            LOG_ERROR("Translator: speech failed (%d)", status);
        }
    }
    leave_voice_phase();
    asr_engine_npu_release();
    xl_translator_listen(XL_LISTEN_SPEECH, 0);
    activity = 0U;
}

static void thread_entry(ULONG argument)
{
    (void)argument;
    open_models();
    for (;;)
    {
        (void)tx_semaphore_get(&wake, TX_WAIT_FOREVER);
        while (take_sentence())
        {
            if (model_ok)
            {
                translate_and_speak();
            }
        }
    }
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

uint32_t xl_translator_create(void)
{
    UINT status = tx_mutex_create(&lock, "xl_translator", TX_INHERIT);

    if (status == TX_SUCCESS)
    {
        status = tx_semaphore_create(&wake, "xl_translator_wake", 0U);
    }
    if (status == TX_SUCCESS)
    {
        status = tx_thread_create(&thread, "xl_translator", thread_entry, 0U, thread_stack, sizeof(thread_stack),
                                  APP_TRANSLATOR_THREAD_PRIORITY, APP_TRANSLATOR_THREAD_PRIORITY, TX_NO_TIME_SLICE,
                                  TX_DONT_START);
    }
    return (uint32_t)status;
}

uint32_t xl_translator_start(void)
{
    return (uint32_t)tx_thread_resume(&thread);
}

void xl_translator_submit(const char *text)
{
    if ((text == NULL) || (text[0] == '\0'))
    {
        return;
    }
    (void)tx_mutex_get(&lock, TX_WAIT_FOREVER);
    if (queue_count == XL_QUEUE_LENGTH)
    {
        queue_head = (queue_head + 1U) % XL_QUEUE_LENGTH; /* drop the oldest */
        queue_count--;
    }
    (void)strncpy(queue[(queue_head + queue_count) % XL_QUEUE_LENGTH], text, XL_SENTENCE_CAPACITY - 1U);
    queue[(queue_head + queue_count) % XL_QUEUE_LENGTH][XL_SENTENCE_CAPACITY - 1U] = '\0';
    queue_count++;
    (void)tx_mutex_put(&lock);
    (void)tx_semaphore_ceiling_put(&wake, 1U);
}

void xl_translator_cancel(void)
{
    (void)tx_mutex_get(&lock, TX_WAIT_FOREVER);
    queue_count = 0U;
    cancel_epoch++;
    (void)tx_mutex_put(&lock);
}

void xl_translator_listen(uint32_t reason, int pause)
{
    (void)tx_mutex_get(&lock, TX_WAIT_FOREVER);
    if (pause)
    {
        listen_reasons |= reason;
    }
    else
    {
        listen_reasons &= ~reason;
    }
    audio_port_set_paused(listen_reasons != 0U);
    (void)tx_mutex_put(&lock);
}

int xl_translator_hindi(char *out, uint32_t capacity, uint32_t *revision)
{
    int changed = 0;

    (void)tx_mutex_get(&lock, TX_WAIT_FOREVER);
    if ((revision != NULL) && (*revision != hindi_revision) && (out != NULL) && (capacity > 0U))
    {
        (void)strncpy(out, hindi, capacity - 1U);
        out[capacity - 1U] = '\0';
        *revision = hindi_revision;
        changed = 1;
    }
    (void)tx_mutex_put(&lock);
    return changed;
}

uint32_t xl_translator_activity(void)
{
    return activity;
}

int xl_translator_model_ok(void)
{
    return model_ok;
}
