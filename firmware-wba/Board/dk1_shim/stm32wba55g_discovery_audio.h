/**
 * @file     stm32wba55g_discovery_audio.h
 * @brief    NUCLEO-WBA55CG stand-in for the DK1 audio BSP: same API, implemented by
 *           Board/nucleo_audio.c for a MAX98357A on SAI1 block B (PB6 SCK, PB5 FS, PB7 SD).
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef NUCLEO_SHIM_STM32WBA55G_DISCOVERY_AUDIO_H
#define NUCLEO_SHIM_STM32WBA55G_DISCOVERY_AUDIO_H

#include "stm32wba55g_discovery.h"

typedef struct
{
  uint32_t Device;        /* Output or input device */
  uint32_t SampleRate;    /* From 8 kHz to 48 kHz */
  uint32_t BitsPerSample; /* 16 */
  uint32_t ChannelsNbr;   /* 1 for mono and 2 for stereo */
  uint32_t Volume;        /* In percentage from 0 to 100 */
} BSP_AUDIO_Init_t;

#define AUDIO_FREQUENCY_48K             48000U
#define AUDIO_FREQUENCY_44K             44100U
#define AUDIO_FREQUENCY_32K             32000U
#define AUDIO_FREQUENCY_24K             24000U
#define AUDIO_FREQUENCY_22K             22050U
#define AUDIO_FREQUENCY_16K             16000U
#define AUDIO_FREQUENCY_8K               8000U

#define AUDIO_OUT_DEVICE_HEADPHONE          1U
#define AUDIO_IN_DEVICE_ANALOG_MIC          0x00U
#define AUDIO_IN_DEVICE_LINE_IN             0x01U
#define AUDIO_IN_DEVICE_DIGITAL_MIC         0x02U

#define AUDIO_RESOLUTION_8B                 8U
#define AUDIO_RESOLUTION_16B               16U
#define AUDIO_RESOLUTION_24B               24U
#define AUDIO_RESOLUTION_32B               32U

#ifndef BSP_AUDIO_OUT_IT_PRIORITY
#define BSP_AUDIO_OUT_IT_PRIORITY          14U
#endif
#ifndef BSP_AUDIO_IN_IT_PRIORITY
#define BSP_AUDIO_IN_IT_PRIORITY           15U
#endif

extern SAI_HandleTypeDef haudio_out_sai;
extern SAI_HandleTypeDef haudio_in_sai;

int32_t BSP_AUDIO_OUT_Init(uint32_t Instance, BSP_AUDIO_Init_t *AudioInit);
int32_t BSP_AUDIO_OUT_DeInit(uint32_t Instance);
int32_t BSP_AUDIO_OUT_Play(uint32_t Instance, uint8_t *pData, uint32_t NbrOfBytes);
int32_t BSP_AUDIO_OUT_Stop(uint32_t Instance);
int32_t BSP_AUDIO_OUT_SetVolume(uint32_t Instance, uint32_t Volume);
void    BSP_AUDIO_OUT_IRQHandler(uint32_t Instance, uint32_t Device);
void    BSP_AUDIO_OUT_TransferComplete_CallBack(uint32_t Instance);
void    BSP_AUDIO_OUT_HalfTransfer_CallBack(uint32_t Instance);
void    BSP_AUDIO_OUT_Error_CallBack(uint32_t Instance);

/* No microphone on the Nucleo: the input side accepts calls and does nothing. */
int32_t BSP_AUDIO_IN_Init(uint32_t Instance, BSP_AUDIO_Init_t *AudioInit);
int32_t BSP_AUDIO_IN_DeInit(uint32_t Instance);
int32_t BSP_AUDIO_IN_Record(uint32_t Instance, uint8_t *pData, uint32_t NbrOfBytes);
int32_t BSP_AUDIO_IN_Stop(uint32_t Instance);
void    BSP_AUDIO_IN_IRQHandler(uint32_t Instance, uint32_t Device);
void    BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance);
void    BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance);

/* Defined by the application (app_entry.c): SAI1 kernel clock from PLL1P. */
HAL_StatusTypeDef MX_SAI1_ClockConfig(SAI_HandleTypeDef *hsai, uint32_t SampleRate);

#endif /* NUCLEO_SHIM_STM32WBA55G_DISCOVERY_AUDIO_H */
