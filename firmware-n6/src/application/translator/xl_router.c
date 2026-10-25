/**
 * @file     xl_router.c
 * @brief    Speaker source selection; implements xl_router.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "xl_router.h"

#include "logger.h"
#include "stm32n6xx_hal.h"
#include "xl_audio_out.h"

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief TMUX1574 EN (active low): Arduino D7. */
#define XL_TMUX_EN_PORT GPIOD
#define XL_TMUX_EN_PIN  GPIO_PIN_6

/** @brief WBA "music playing" line, through the TMUX: Arduino D2. */
#define XL_MUSIC_PORT GPIOD
#define XL_MUSIC_PIN  GPIO_PIN_0

/** @brief The music line must be high this long before music counts as playing. */
#define XL_MUSIC_ON_MS 100U

/** @brief ... and low this long before it counts as stopped (gaps between songs). */
#define XL_MUSIC_OFF_MS 1500U

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

static xl_mode_t mode = XL_MODE_AUTO;
static xl_source_t source = XL_SOURCE_WBA;
static xl_source_t target = XL_SOURCE_WBA;
static int music;
static int music_raw;
static uint32_t music_raw_since;
static int initialized;

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Source the rules want now. */
static xl_source_t wanted(void)
{
    if (mode == XL_MODE_MUSIC)
    {
        return XL_SOURCE_WBA;
    }
    if (mode == XL_MODE_TRANSLATE)
    {
        return XL_SOURCE_N6;
    }
    return music ? XL_SOURCE_WBA : XL_SOURCE_N6;
}

/** @brief Debounce the music line; 1 when the debounced state changed. */
static int read_music(uint32_t now_ms)
{
    const int raw = (HAL_GPIO_ReadPin(XL_MUSIC_PORT, XL_MUSIC_PIN) == GPIO_PIN_SET);

    if (raw != music_raw)
    {
        music_raw = raw;
        music_raw_since = now_ms;
    }
    if (raw && !music && ((now_ms - music_raw_since) >= XL_MUSIC_ON_MS))
    {
        music = 1;
        LOG_INFO("Router: WBA music started");
        return 1;
    }
    if (!raw && music && ((now_ms - music_raw_since) >= XL_MUSIC_OFF_MS))
    {
        music = 0;
        LOG_INFO("Router: WBA music stopped");
        return 1;
    }
    return 0;
}

static xl_source_t source_of(xl_audio_source_t audio)
{
    return (audio == XL_AUDIO_SPEECH) ? XL_SOURCE_N6 : XL_SOURCE_WBA;
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void xl_router_init(void)
{
    GPIO_InitTypeDef gpio = {0};

    __HAL_RCC_GPIOD_CLK_ENABLE();

    /* The TMUX stays open (EN high) until the receive pins are set up. */
    HAL_GPIO_WritePin(XL_TMUX_EN_PORT, XL_TMUX_EN_PIN, GPIO_PIN_SET);
    gpio.Pin = XL_TMUX_EN_PIN;
    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(XL_TMUX_EN_PORT, &gpio);

    gpio.Pin = XL_MUSIC_PIN;
    gpio.Mode = GPIO_MODE_INPUT;
    gpio.Pull = GPIO_PULLDOWN; /* no WBA connected or powered, or the TMUX open: no music */
    HAL_GPIO_Init(XL_MUSIC_PORT, &gpio);

    source = XL_SOURCE_WBA;
    target = XL_SOURCE_WBA;
    xl_audio_out_select(XL_AUDIO_MUSIC);
    music = 0;
    music_raw = 0;
    music_raw_since = 0U;
    initialized = 1;
}

void xl_router_connect_wba(int connect)
{
    HAL_GPIO_WritePin(XL_TMUX_EN_PORT, XL_TMUX_EN_PIN, connect ? GPIO_PIN_RESET : GPIO_PIN_SET);
    LOG_INFO("Router: TMUX1574 %s (D7 %s)", connect ? "closed, WBA connected" : "open", connect ? "low" : "high");
}

void xl_router_set_mode(xl_mode_t new_mode)
{
    if (new_mode != mode)
    {
        static const char *const names[] = { "AUTO", "TRANSLATE", "MUSIC" };

        mode = new_mode;
        LOG_INFO("Router: mode %s", names[mode]);
    }
}

xl_mode_t xl_router_mode(void)
{
    return mode;
}

int xl_router_music(void)
{
    return music;
}

xl_source_t xl_router_source(void)
{
    return source;
}

xl_source_t xl_router_target(void)
{
    return target;
}

int xl_router_poll(uint32_t now_ms)
{
    int changed;
    xl_source_t now_playing;

    if (!initialized)
    {
        return 0;
    }
    changed = read_music(now_ms);
    xl_audio_out_poll(now_ms);
    if (wanted() != target)
    {
        target = wanted();
        xl_audio_out_select((target == XL_SOURCE_N6) ? XL_AUDIO_SPEECH : XL_AUDIO_MUSIC);
    }
    now_playing = source_of(xl_audio_out_selected());
    if (now_playing != source)
    {
        source = now_playing;
        changed = 1;
        LOG_INFO("Router: the speaker plays the %s", (source == XL_SOURCE_N6) ? "N6 translator" : "WBA music");
    }
    return changed;
}
