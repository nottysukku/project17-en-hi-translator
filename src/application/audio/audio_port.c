/**
 * @file     audio_port.c
 * @brief    Microphone as the asr_engine audio port; implements audio_port.h.
 * @author   Nikita Prismakov
 *
 * SPDX-FileCopyrightText: 2026 Nikita Prismakov
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "audio_port.h"

#include <stddef.h>

#include "asr_engine.h"
#include "bsp_mic.h"

#ifdef APP_AUDIO_CAPTURE
#include "capture.h"
#endif

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

/** @brief Non-zero while recognition is paused (audio_port_set_paused()). */
static volatile int paused;

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Wake the conversion thread once per completed capture chunk, unless paused. */
static void chunk_ready_callback(void *context)
{
  (void)context;
  if (!paused)
  {
    asr_engine_audio_chunk_ready();
  }
}

/** @brief Forward to drv_pdm_available_chunks(); none while paused. */
static uint32_t port_available_chunks(void *context)
{
  (void)context;
  return paused ? 0U : drv_pdm_available_chunks();
}

/** @brief Forward to drv_pdm_read(), with its status as the driver-agnostic asr_audio_status_t. */
static asr_audio_status_t port_read(void *context,
                                    int16_t *destination,
                                    uint32_t destination_samples,
                                    uint32_t chunks)
{
  (void)context;
  switch (drv_pdm_read(destination, destination_samples, chunks))
  {
  case DRV_PDM_OK:
#ifdef APP_AUDIO_CAPTURE
    /* Exactly the samples the recognizer is about to process. */
    capture_push_audio(destination, chunks * ASR_MODEL_MEL_HOP_SAMPLES);
#endif
    return ASR_AUDIO_OK;
  case DRV_PDM_UNDERRUN:
    return ASR_AUDIO_UNDERRUN;
  case DRV_PDM_OVERRUN:
    return ASR_AUDIO_OVERRUN;
  default:
    return ASR_AUDIO_ERROR;
  }
}

/** @brief Forward to drv_pdm_take_dropped_chunks(). */
static uint32_t port_take_dropped_chunks(void *context)
{
  (void)context;
  return drv_pdm_take_dropped_chunks();
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

uint32_t audio_port_start(void)
{
  drv_pdm_config_t config = {0};

  config.chunk_samples = ASR_MODEL_MEL_HOP_SAMPLES;
  config.chunk_count = ASR_CONF_CAPTURE_CHUNK_COUNT;
  config.chunk_callback = chunk_ready_callback;
  config.callback_context = NULL;
  return (uint32_t)bsp_mic_start(&config);
}

void audio_port_set_paused(int pause)
{
  if (!pause && paused)
  {
    /* Drop what was captured while paused: it is stale, and the ring overflowed meanwhile. */
    uint32_t chunks;

    (void)drv_pdm_take_dropped_chunks();
    while ((chunks = drv_pdm_available_chunks()) > 0U)
    {
      if (drv_pdm_skip(chunks) != DRV_PDM_OK)
      {
        break;
      }
    }
  }
  paused = (pause != 0);
}

const asr_audio_port_t *audio_port_get(void)
{
  /** @brief The microphone's plug into the model layer; the PDM driver is a singleton. */
  static const asr_audio_port_t mic_port = {
    .context = NULL,
    .available_chunks = port_available_chunks,
    .read = port_read,
    .take_dropped_chunks = port_take_dropped_chunks,
  };

  return &mic_port;
}
