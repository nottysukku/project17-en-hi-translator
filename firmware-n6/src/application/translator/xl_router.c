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

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief TMUX1574 SEL: Arduino D7. */
#define XL_SEL_PORT GPIOD
#define XL_SEL_PIN  GPIO_PIN_6

/** @brief MAX98357A SD (shutdown), open-drain: Arduino D4. */
#define XL_AMP_SD_PORT GPIOH
#define XL_AMP_SD_PIN  GPIO_PIN_5

/** @brief WBA "music playing" line: Arduino D2. */
#define XL_MUSIC_PORT GPIOD
#define XL_MUSIC_PIN  GPIO_PIN_0

/** @brief The music line must be high this long before music counts as playing. */
#define XL_MUSIC_ON_MS 100U

/** @brief ... and low this long before it counts as stopped (gaps between songs). */
#define XL_MUSIC_OFF_MS 1500U

/** @brief Amplifier muted this long before and after SEL changes. */
#define XL_SWITCH_MUTE_MS 20U

/*******************************************************************************/
/*                                 PRIVATE TYPES                               */
/*******************************************************************************/

/** @brief Steps of a source switch. */
typedef enum
{
    SWITCH_IDLE = 0, /**< Source settled, amplifier on. */
    SWITCH_MUTED,    /**< Amplifier off, SEL about to change. */
    SWITCH_SETTLING  /**< SEL changed, amplifier about to come back. */
} switch_step_t;

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

static xl_mode_t mode = XL_MODE_AUTO;
static xl_source_t source = XL_SOURCE_WBA;
static xl_source_t target = XL_SOURCE_WBA;
static switch_step_t step = SWITCH_IDLE;
static uint32_t step_at;
static int music;
static int music_raw;
static uint32_t music_raw_since;
static int initialized;

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Drive SEL for @p to. */
static void set_sel(xl_source_t to)
{
    HAL_GPIO_WritePin(XL_SEL_PORT, XL_SEL_PIN, (to == XL_SOURCE_N6) ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/** @brief Shut the amplifier down (pull SD low) or release it. */
static void set_amp_on(int on)
{
    /* Open drain: "set" releases the line, the module's bias takes over. */
    HAL_GPIO_WritePin(XL_AMP_SD_PORT, XL_AMP_SD_PIN, on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

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

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void xl_router_init(void)
{
    GPIO_InitTypeDef gpio = {0};

    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();

    /* Start where the hardware already is: WBA selected, amplifier on. */
    set_sel(XL_SOURCE_WBA);
    gpio.Pin = XL_SEL_PIN;
    gpio.Mode = GPIO_MODE_OUTPUT_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(XL_SEL_PORT, &gpio);

    set_amp_on(1);
    gpio.Pin = XL_AMP_SD_PIN;
    gpio.Mode = GPIO_MODE_OUTPUT_OD;
    HAL_GPIO_Init(XL_AMP_SD_PORT, &gpio);

    gpio.Pin = XL_MUSIC_PIN;
    gpio.Mode = GPIO_MODE_INPUT;
    gpio.Pull = GPIO_PULLDOWN; /* no WBA connected or powered: no music */
    HAL_GPIO_Init(XL_MUSIC_PORT, &gpio);

    source = XL_SOURCE_WBA;
    target = XL_SOURCE_WBA;
    step = SWITCH_IDLE;
    music = 0;
    music_raw = 0;
    music_raw_since = 0U;
    initialized = 1;
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

    if (!initialized)
    {
        return 0;
    }
    changed = read_music(now_ms);
    switch (step)
    {
    case SWITCH_IDLE:
        target = wanted();
        if (target != source)
        {
            set_amp_on(0);
            step = SWITCH_MUTED;
            step_at = now_ms;
        }
        break;
    case SWITCH_MUTED:
        if ((now_ms - step_at) >= XL_SWITCH_MUTE_MS)
        {
            set_sel(target);
            source = target;
            step = SWITCH_SETTLING;
            step_at = now_ms;
            changed = 1;
            LOG_INFO("Router: speaker now plays the %s", (source == XL_SOURCE_N6) ? "N6 translator" : "WBA music");
        }
        break;
    case SWITCH_SETTLING:
    default:
        if ((now_ms - step_at) >= XL_SWITCH_MUTE_MS)
        {
            set_amp_on(1);
            step = SWITCH_IDLE;
        }
        break;
    }
    return changed;
}
