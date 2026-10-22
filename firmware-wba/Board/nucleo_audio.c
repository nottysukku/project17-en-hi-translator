/**
 * @file     nucleo_audio.c
 * @brief    Audio output of the NUCLEO-WBA55CG music receiver: SAI1 block B as I2S master into a
 *           MAX98357A (through the Project 17 TMUX1574 switch).
 * @details  Implements the subset of the STM32WBA55G-DK1 audio BSP that ST's TMAP application calls,
 *           with the same SAI frame as the DK1 (I2S Philips, 16-bit slots, BCLK = 32 x fs) and the
 *           same pins: PB6 SCK (Arduino D3), PB5 FS (morpho CN4-35), PB7 SD (Arduino D2). The DK1
 *           driver also programmed a CS42L51 codec over I2C and drove MCLK on PB4; the MAX98357A
 *           needs neither. Volume is applied in software by music_link.c, as the amplifier has
 *           no volume control. There is no microphone: the input functions do nothing.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "stm32wba55g_discovery_audio.h"

#include "music_link.h"

/* SAI1 block B, pins and DMA channel as on the STM32WBA55G-DK1. */
#define OUT_SAI                 SAI1_Block_B
#define OUT_PORT                GPIOB
#define OUT_PINS                (GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7)
#define OUT_AF                  GPIO_AF13_SAI1
#define OUT_DMA_CHANNEL         GPDMA1_Channel2
#define OUT_DMA_REQUEST         GPDMA1_REQUEST_SAI1_B
#define OUT_DMA_IRQ             GPDMA1_Channel2_IRQn

SAI_HandleTypeDef haudio_out_sai = {0};
SAI_HandleTypeDef haudio_in_sai = {0};
DMA_HandleTypeDef hDmaSaiTx = {0};

static DMA_QListTypeDef tx_queue;
static DMA_NodeTypeDef tx_node;

/* RESET, STOP or PLAYING, as the DK1 driver's states. */
enum { OUT_RESET = 0, OUT_PLAYING = 1, OUT_STOP = 2 };
static uint32_t out_state = OUT_RESET;

/** @brief Pins, the circular DMA node and its interrupt. */
static int32_t out_msp_init(SAI_HandleTypeDef *hsai)
{
  GPIO_InitTypeDef gpio = {0};
  DMA_NodeConfTypeDef node = {0};

  __HAL_RCC_SAI1_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  gpio.Pin = OUT_PINS;
  gpio.Mode = GPIO_MODE_AF_PP;
  gpio.Pull = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_HIGH;
  gpio.Alternate = OUT_AF;
  HAL_GPIO_Init(OUT_PORT, &gpio);

  __HAL_RCC_GPDMA1_CLK_ENABLE();
  if (tx_queue.Head != NULL)
  {
    (void)HAL_DMAEx_List_DeInit(&hDmaSaiTx);
    (void)HAL_DMAEx_List_ResetQ(&tx_queue);
  }
  node.NodeType = DMA_GPDMA_LINEAR_NODE;
  node.Init.Request = OUT_DMA_REQUEST;
  node.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
  node.Init.Direction = DMA_MEMORY_TO_PERIPH;
  node.Init.SrcInc = DMA_SINC_INCREMENTED;
  node.Init.DestInc = DMA_DINC_FIXED;
  node.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_HALFWORD;
  node.Init.DestDataWidth = DMA_DEST_DATAWIDTH_HALFWORD;
  node.Init.SrcBurstLength = 1;
  node.Init.DestBurstLength = 1;
  node.Init.Priority = DMA_HIGH_PRIORITY;
  node.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
  node.Init.TransferAllocatedPort = DMA_SRC_ALLOCATED_PORT0 | DMA_DEST_ALLOCATED_PORT1;
  node.DataHandlingConfig.DataExchange = DMA_EXCHANGE_NONE;
  node.DataHandlingConfig.DataAlignment = DMA_DATA_RIGHTALIGN_ZEROPADDED;
  node.TriggerConfig.TriggerPolarity = DMA_TRIG_POLARITY_MASKED;
  if ((HAL_DMAEx_List_BuildNode(&node, &tx_node) != HAL_OK) ||
      (HAL_DMAEx_List_InsertNode_Tail(&tx_queue, &tx_node) != HAL_OK) ||
      (HAL_DMAEx_List_SetCircularMode(&tx_queue) != HAL_OK))
  {
    return BSP_ERROR_PERIPH_FAILURE;
  }

  hDmaSaiTx.Instance = OUT_DMA_CHANNEL;
  hDmaSaiTx.InitLinkedList.Priority = DMA_HIGH_PRIORITY;
  hDmaSaiTx.InitLinkedList.LinkStepMode = DMA_LSM_FULL_EXECUTION;
  hDmaSaiTx.InitLinkedList.LinkAllocatedPort = DMA_LINK_ALLOCATED_PORT1;
  hDmaSaiTx.InitLinkedList.TransferEventMode = DMA_TCEM_LAST_LL_ITEM_TRANSFER;
  hDmaSaiTx.InitLinkedList.LinkedListMode = DMA_LINKEDLIST_CIRCULAR;
  if ((HAL_DMAEx_List_Init(&hDmaSaiTx) != HAL_OK) || (HAL_DMAEx_List_LinkQ(&hDmaSaiTx, &tx_queue) != HAL_OK))
  {
    return BSP_ERROR_PERIPH_FAILURE;
  }
  __HAL_LINKDMA(hsai, hdmatx, hDmaSaiTx);
  HAL_NVIC_SetPriority(OUT_DMA_IRQ, BSP_AUDIO_OUT_IT_PRIORITY, 0);
  HAL_NVIC_EnableIRQ(OUT_DMA_IRQ);
  return BSP_ERROR_NONE;
}

/** @brief I2S Philips framing as the DK1 driver: 16-bit slots, FS low for the left slot. */
static int32_t out_sai_init(SAI_HandleTypeDef *hsai, const BSP_AUDIO_Init_t *init)
{
  __HAL_SAI_DISABLE(hsai);
  hsai->Init.MonoStereoMode = (init->ChannelsNbr == 1U) ? SAI_MONOMODE : SAI_STEREOMODE;
  hsai->Init.AudioFrequency = init->SampleRate;
  hsai->Init.AudioMode = SAI_MODEMASTER_TX;
  hsai->Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
  hsai->Init.Protocol = SAI_FREE_PROTOCOL;
  hsai->Init.DataSize = SAI_DATASIZE_16;
  hsai->Init.FirstBit = SAI_FIRSTBIT_MSB;
  hsai->Init.ClockStrobing = SAI_CLOCKSTROBING_FALLINGEDGE;
  hsai->Init.Synchro = SAI_ASYNCHRONOUS;
  hsai->Init.OutputDrive = SAI_OUTPUTDRIVE_ENABLE;
  hsai->Init.FIFOThreshold = SAI_FIFOTHRESHOLD_1QF;
  hsai->Init.SynchroExt = SAI_SYNCEXT_DISABLE;
  hsai->Init.CompandingMode = SAI_NOCOMPANDING;
  hsai->Init.TriState = SAI_OUTPUT_NOTRELEASED;
  hsai->Init.Mckdiv = 0U;
  hsai->Init.MckOutput = SAI_MCK_OUTPUT_DISABLE; /* the MAX98357A runs without MCLK */
  hsai->Init.MckOverSampling = SAI_MCK_OVERSAMPLING_DISABLE;
  hsai->Init.PdmInit.Activation = DISABLE;
  hsai->FrameInit.FrameLength = 32;
  hsai->FrameInit.ActiveFrameLength = 16;
  hsai->FrameInit.FSDefinition = SAI_FS_CHANNEL_IDENTIFICATION;
  hsai->FrameInit.FSPolarity = SAI_FS_ACTIVE_LOW;
  hsai->FrameInit.FSOffset = SAI_FS_BEFOREFIRSTBIT;
  hsai->SlotInit.FirstBitOffset = 0;
  hsai->SlotInit.SlotSize = SAI_SLOTSIZE_16B;
  hsai->SlotInit.SlotNumber = 2;
  hsai->SlotInit.SlotActive = SAI_SLOTACTIVE_0 | SAI_SLOTACTIVE_1;
  return (HAL_SAI_Init(hsai) == HAL_OK) ? BSP_ERROR_NONE : BSP_ERROR_PERIPH_FAILURE;
}

int32_t BSP_AUDIO_OUT_Init(uint32_t Instance, BSP_AUDIO_Init_t *AudioInit)
{
  if ((Instance != 0U) || (AudioInit == NULL) || (AudioInit->BitsPerSample != AUDIO_RESOLUTION_16B))
  {
    return BSP_ERROR_WRONG_PARAM;
  }
  if (out_state != OUT_RESET)
  {
    return BSP_ERROR_BUSY;
  }
  haudio_out_sai.Instance = OUT_SAI;
  if (MX_SAI1_ClockConfig(&haudio_out_sai, AudioInit->SampleRate) != HAL_OK)
  {
    return BSP_ERROR_CLOCK_FAILURE;
  }
  if (out_msp_init(&haudio_out_sai) != BSP_ERROR_NONE)
  {
    return BSP_ERROR_PERIPH_FAILURE;
  }
  if (out_sai_init(&haudio_out_sai, AudioInit) != BSP_ERROR_NONE)
  {
    return BSP_ERROR_PERIPH_FAILURE;
  }
  MUSIC_LINK_SetVolume(AudioInit->Volume);
  out_state = OUT_STOP;
  return BSP_ERROR_NONE;
}

int32_t BSP_AUDIO_OUT_DeInit(uint32_t Instance)
{
  if (Instance != 0U)
  {
    return BSP_ERROR_WRONG_PARAM;
  }
  if (out_state == OUT_RESET)
  {
    return BSP_ERROR_NONE;
  }
  MUSIC_LINK_StreamStopped();
  (void)HAL_SAI_DeInit(&haudio_out_sai);
  HAL_NVIC_DisableIRQ(OUT_DMA_IRQ);
  (void)HAL_DMAEx_List_DeInit(&hDmaSaiTx);
  (void)HAL_DMAEx_List_ResetQ(&tx_queue);
  HAL_GPIO_DeInit(OUT_PORT, OUT_PINS);
  out_state = OUT_RESET;
  return BSP_ERROR_NONE;
}

int32_t BSP_AUDIO_OUT_Play(uint32_t Instance, uint8_t *pData, uint32_t NbrOfBytes)
{
  if ((Instance != 0U) || (pData == NULL))
  {
    return BSP_ERROR_WRONG_PARAM;
  }
  if (out_state != OUT_STOP)
  {
    return BSP_ERROR_BUSY;
  }
  if (HAL_SAI_Transmit_DMA(&haudio_out_sai, pData, (uint16_t)(NbrOfBytes / 2U)) != HAL_OK)
  {
    return BSP_ERROR_PERIPH_FAILURE;
  }
  out_state = OUT_PLAYING;
  return BSP_ERROR_NONE;
}

int32_t BSP_AUDIO_OUT_Stop(uint32_t Instance)
{
  if (Instance != 0U)
  {
    return BSP_ERROR_WRONG_PARAM;
  }
  MUSIC_LINK_StreamStopped();
  if (out_state != OUT_PLAYING)
  {
    return BSP_ERROR_NONE;
  }
  if (HAL_SAI_DMAStop(&haudio_out_sai) != HAL_OK)
  {
    return BSP_ERROR_PERIPH_FAILURE;
  }
  out_state = OUT_STOP;
  return BSP_ERROR_NONE;
}

int32_t BSP_AUDIO_OUT_SetVolume(uint32_t Instance, uint32_t Volume)
{
  if (Instance != 0U)
  {
    return BSP_ERROR_WRONG_PARAM;
  }
  MUSIC_LINK_SetVolume(Volume);
  return BSP_ERROR_NONE;
}

void BSP_AUDIO_OUT_IRQHandler(uint32_t Instance, uint32_t Device)
{
  UNUSED(Instance);
  UNUSED(Device);
  HAL_DMA_IRQHandler(haudio_out_sai.hdmatx);
}

void HAL_SAI_TxCpltCallback(SAI_HandleTypeDef *hsai)
{
  UNUSED(hsai);
  BSP_AUDIO_OUT_TransferComplete_CallBack(0);
}

void HAL_SAI_TxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
  UNUSED(hsai);
  BSP_AUDIO_OUT_HalfTransfer_CallBack(0);
}

void HAL_SAI_ErrorCallback(SAI_HandleTypeDef *hsai)
{
  if (hsai->Instance == OUT_SAI)
  {
    BSP_AUDIO_OUT_Error_CallBack(0);
  }
}

__weak void BSP_AUDIO_OUT_TransferComplete_CallBack(uint32_t Instance)
{
  UNUSED(Instance);
}

__weak void BSP_AUDIO_OUT_HalfTransfer_CallBack(uint32_t Instance)
{
  UNUSED(Instance);
}

__weak void BSP_AUDIO_OUT_Error_CallBack(uint32_t Instance)
{
  UNUSED(Instance);
}

/* ---- Input: no microphone on this board. ------------------------------------------------------ */

int32_t BSP_AUDIO_IN_Init(uint32_t Instance, BSP_AUDIO_Init_t *AudioInit)
{
  UNUSED(Instance);
  UNUSED(AudioInit);
  /* Start_RxAudio() and MX_AudioDeInit() touch this handle's registers: give it a real block
     that is never enabled. */
  haudio_in_sai.Instance = SAI1_Block_A;
  return BSP_ERROR_NONE;
}

int32_t BSP_AUDIO_IN_DeInit(uint32_t Instance)
{
  UNUSED(Instance);
  return BSP_ERROR_NONE;
}

int32_t BSP_AUDIO_IN_Record(uint32_t Instance, uint8_t *pData, uint32_t NbrOfBytes)
{
  UNUSED(Instance);
  UNUSED(pData);
  UNUSED(NbrOfBytes);
  return BSP_ERROR_NONE;
}

int32_t BSP_AUDIO_IN_Stop(uint32_t Instance)
{
  UNUSED(Instance);
  return BSP_ERROR_NONE;
}

void BSP_AUDIO_IN_IRQHandler(uint32_t Instance, uint32_t Device)
{
  UNUSED(Instance);
  UNUSED(Device);
}

__weak void BSP_AUDIO_IN_TransferComplete_CallBack(uint32_t Instance)
{
  UNUSED(Instance);
}

__weak void BSP_AUDIO_IN_HalfTransfer_CallBack(uint32_t Instance)
{
  UNUSED(Instance);
}
