/**
 * @file     music_link.c
 * @brief    Music level and the "music playing" line; implements music_link.h.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "music_link.h"

#include <math.h>

#include "stm32wbaxx_hal.h"
#include "stm32_timer.h"

#define LINE_PORT GPIOB
#define LINE_PIN  GPIO_PIN_0

/** @brief Gain applied to every sample, Q15. */
static volatile int32_t gain_q15 = 0;

/** @brief Decoded channel frames seen; an audible one since the last poll. */
static volatile uint32_t frames_seen;
static volatile uint32_t audible_since_poll;

/** @brief Frame count at the previous poll; polls since the last audible frame. */
static uint32_t frames_at_poll;
static uint32_t polls_since_audible = 0xFFFFU;

static UTIL_TIMER_Object_t poll_timer;

/** @brief Drive the line; the pin mode is set again each time, in case a low-power exit reset it. */
static void set_line(int on)
{
  HAL_GPIO_WritePin(LINE_PORT, LINE_PIN, on ? GPIO_PIN_SET : GPIO_PIN_RESET);
  if ((LINE_PORT->MODER & (3U << (2U * 0U))) != (1U << (2U * 0U)))
  {
    MODIFY_REG(LINE_PORT->MODER, 3U << (2U * 0U), 1U << (2U * 0U)); /* PB0: general-purpose output */
  }
}

/** @brief Every MUSIC_LINK_POLL_MS: music plays while frames arrive and one was audible lately. */
static void poll(void *arg)
{
  const uint32_t seen = frames_seen;
  const int streaming = (seen != frames_at_poll);

  (void)arg;
  frames_at_poll = seen;
  if (audible_since_poll)
  {
    audible_since_poll = 0U;
    polls_since_audible = 0U;
  }
  else if (polls_since_audible < 0xFFFFU)
  {
    ++polls_since_audible;
  }
  set_line(streaming && ((polls_since_audible * MUSIC_LINK_POLL_MS) < MUSIC_LINK_SILENT_MS));
}

void MUSIC_LINK_Init(void)
{
  GPIO_InitTypeDef gpio = {0};

  __HAL_RCC_GPIOB_CLK_ENABLE();
  set_line(0);
  gpio.Pin = LINE_PIN;
  gpio.Mode = GPIO_MODE_OUTPUT_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LINE_PORT, &gpio);
  set_line(0);

  MUSIC_LINK_SetVolume(50U);
  (void)UTIL_TIMER_Create(&poll_timer, MUSIC_LINK_POLL_MS, UTIL_TIMER_PERIODIC, poll, NULL);
  (void)UTIL_TIMER_Start(&poll_timer);
}

void MUSIC_LINK_SetVolume(uint32_t percent)
{
  if (percent == 0U)
  {
    gain_q15 = 0;
    return;
  }
  if (percent > 100U)
  {
    percent = 100U;
  }
  {
    const float db = -MUSIC_LINK_RANGE_DB * (float)(100U - percent) / 100.0f;
    const float g = (float)MUSIC_LINK_CEILING_Q15 * powf(10.0f, db / 20.0f);

    gain_q15 = (int32_t)(g + 0.5f);
  }
}

void MUSIC_LINK_ProcessChannel(int16_t *samples, uint32_t frames)
{
  const int32_t g = gain_q15;
  int32_t peak = 0;
  uint32_t i;

  for (i = 0U; i < frames; ++i)
  {
    const int32_t s = samples[2U * i];
    const int32_t a = (s < 0) ? -s : s;

    if (a > peak)
    {
      peak = a;
    }
    samples[2U * i] = (int16_t)((s * g) >> 15);
  }
  ++frames_seen;
  if ((peak >= MUSIC_LINK_AUDIBLE_PEAK) && (g > 0))
  {
    audible_since_poll = 1U;
  }
}

void MUSIC_LINK_StreamStopped(void)
{
  audible_since_poll = 0U;
  polls_since_audible = 0xFFFFU;
  set_line(0);
}
