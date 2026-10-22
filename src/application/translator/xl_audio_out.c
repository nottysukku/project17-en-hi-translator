/**
 * @file     xl_audio_out.c
 * @brief    Translator speaker output; implements xl_audio_out.h.
 * @details  GPDMA1 channel 7 runs one circular node over a stereo double buffer of
 *           XL_AUDIO_HALF_FRAMES frames per half (about 5 ms). The SAI half- and full-transfer
 *           callbacks refill the half just played from the mono ring. The double buffer and the
 *           node live in the non-cacheable section, as the microphone's capture area does
 *           (drv_pdm_dma.c), so no cache maintenance is needed; the ring is only touched by the
 *           CPU and lives in external PSRAM.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "xl_audio_out.h"

#include <math.h>
#include <string.h>

#include "mcu_clock.h"
#include "stm32n6xx_hal.h"
#include "stm32n6xx_ll_rcc.h"

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief Stereo frames per DMA half (5.3 ms): latency and interrupt rate. */
#define XL_AUDIO_HALF_FRAMES 256U

/** @brief Samples (both slots) of the whole double buffer. */
#define XL_AUDIO_DMA_SAMPLES (2U * XL_AUDIO_HALF_FRAMES * 2U)

/** @brief Mono samples the ring holds (4 s); one slot stays empty to tell full from empty. */
#define XL_AUDIO_RING_SAMPLES (4U * XL_AUDIO_RATE_HZ)

/** @brief NVIC priority of the DMA interrupt; it calls no RTOS service. */
#define XL_AUDIO_IRQ_PRIORITY 6U

/** @brief Entries of the sine table used by xl_audio_out_tone(). */
#define XL_AUDIO_SINE_ENTRIES 256U

/** @brief Fade-in and fade-out of a tone, in samples (5 ms). */
#define XL_AUDIO_FADE_SAMPLES (XL_AUDIO_RATE_HZ / 200U)

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

/** @brief HAL handle of SAI2 block B. */
static SAI_HandleTypeDef sai;

/** @brief HAL handle of GPDMA1 channel 7. */
static DMA_HandleTypeDef dma;

/** @brief Linked-list queue of the channel: one circular node. */
static DMA_QListTypeDef queue;

/** @brief The node, read by the DMA; non-cacheable like the buffer it points at. */
static DMA_NodeTypeDef node __NON_CACHEABLE __attribute__((aligned(32)));

/** @brief Stereo double buffer streamed to SAI2 (L, R, L, R, ...). */
static int16_t dma_buffer[XL_AUDIO_DMA_SAMPLES] __NON_CACHEABLE __attribute__((aligned(32)));

/** @brief Mono samples waiting to be played; written by the application, read by the interrupt. */
static int16_t ring[XL_AUDIO_RING_SAMPLES] __attribute__((section(".psram_bss"), aligned(32)));

/** @brief Next ring slot to write; only the writer changes it. */
static volatile uint32_t ring_head;

/** @brief Next ring slot to play; only the DMA interrupt changes it (flush masks interrupts). */
static volatile uint32_t ring_tail;

/** @brief DMA halves that found the ring empty while samples were expected. */
static volatile uint32_t underruns;

/** @brief Non-zero while the stream runs. */
static volatile int running;

/** @brief Non-zero once xl_audio_out_init() succeeded. */
static int initialized;

/** @brief One sine period, for test tones. */
static int16_t sine[XL_AUDIO_SINE_ENTRIES];

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Ring index after @p index. */
static uint32_t ring_next(uint32_t index)
{
    return ((index + 1U) == XL_AUDIO_RING_SAMPLES) ? 0U : (index + 1U);
}

/**
 * @brief Refill one DMA half from the ring; runs in the DMA interrupt.
 * @param[out] out First stereo sample of the half.
 */
static void fill_half(int16_t *out)
{
    uint32_t tail = ring_tail;
    const uint32_t head = ring_head;
    uint32_t frame;
    int short_of_samples = 0;

    for (frame = 0U; frame < XL_AUDIO_HALF_FRAMES; ++frame)
    {
        int16_t sample = 0;

        if (tail != head)
        {
            sample = ring[tail];
            tail = ring_next(tail);
        }
        else if (frame > 0U)
        {
            short_of_samples = 1; /* the ring ran dry in the middle of this half */
        }
        out[2U * frame] = sample;
        out[(2U * frame) + 1U] = sample;
    }
    ring_tail = tail;
    if (short_of_samples)
    {
        ++underruns;
    }
}

/**
 * @brief Make sure IC8 runs from PLL3 and select it as the SAI2 kernel clock.
 * @return 0 on success, negative on failure.
 * @note The microphone (drv_pdm_mdf.c) configures the same: PLL3 / 1. IC8 is only programmed
 *       here if it is not running yet, so a running microphone is never disturbed.
 */
static int init_clock(void)
{
    if ((LL_RCC_PLL3_IsReady() == 0U) && !mcu_clock_enable_audio_pll())
    {
        return -1;
    }
    if (LL_RCC_IC8_IsEnabled() == 0U)
    {
        LL_RCC_IC8_SetSource(LL_RCC_ICCLKSOURCE_PLL3);
        LL_RCC_IC8_SetDivider(1U);
        LL_RCC_IC8_Enable();
    }
    __HAL_RCC_SAI2_CONFIG(RCC_SAI2CLKSOURCE_IC8);
    __HAL_RCC_SAI2_CLK_ENABLE();
    return 0;
}

/** @brief PE12 SCK, PE13 FS, PE7 SD as SAI2 block B (AF10), secure like the microphone pins. */
static void init_pins(void)
{
    GPIO_InitTypeDef gpio = {0};

    __HAL_RCC_GPIOE_CLK_ENABLE();
    gpio.Pin = GPIO_PIN_7 | GPIO_PIN_12 | GPIO_PIN_13;
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    gpio.Alternate = GPIO_AF10_SAI2;
    HAL_GPIO_Init(GPIOE, &gpio);
    HAL_GPIO_ConfigPinAttributes(GPIOE, GPIO_PIN_7, GPIO_PIN_SEC | GPIO_PIN_NPRIV);
    HAL_GPIO_ConfigPinAttributes(GPIOE, GPIO_PIN_12, GPIO_PIN_SEC | GPIO_PIN_NPRIV);
    HAL_GPIO_ConfigPinAttributes(GPIOE, GPIO_PIN_13, GPIO_PIN_SEC | GPIO_PIN_NPRIV);
}

/**
 * @brief One circular node over the double buffer into SAI2_B->DR, on GPDMA1 channel 7.
 * @return 0 on success, negative on failure.
 * @note Port allocation as the microphone's channel: memory on port 1, peripheral on port 0.
 */
static int init_dma(void)
{
    DMA_NodeConfTypeDef config = {0};

    __HAL_RCC_GPDMA1_CLK_ENABLE();
    config.NodeType = DMA_GPDMA_LINEAR_NODE;
    config.Init.Request = GPDMA1_REQUEST_SAI2_B;
    config.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
    config.Init.Direction = DMA_MEMORY_TO_PERIPH;
    config.Init.SrcInc = DMA_SINC_INCREMENTED;
    config.Init.DestInc = DMA_DINC_FIXED;
    config.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_HALFWORD;
    config.Init.DestDataWidth = DMA_DEST_DATAWIDTH_HALFWORD;
    config.Init.SrcBurstLength = 1U;
    config.Init.DestBurstLength = 1U;
    config.Init.TransferAllocatedPort = DMA_SRC_ALLOCATED_PORT1 | DMA_DEST_ALLOCATED_PORT0;
    config.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
    config.Init.Mode = DMA_NORMAL;
    config.TriggerConfig.TriggerPolarity = DMA_TRIG_POLARITY_MASKED;
    config.DataHandlingConfig.DataExchange = DMA_EXCHANGE_NONE;
    config.DataHandlingConfig.DataAlignment = DMA_DATA_RIGHTALIGN_ZEROPADDED;
    config.SrcSecure = DMA_CHANNEL_SRC_SEC;
    config.DestSecure = DMA_CHANNEL_DEST_SEC;
    config.SrcAddress = (uint32_t)dma_buffer;
    config.DstAddress = (uint32_t)&SAI2_Block_B->DR;
    config.DataSize = sizeof(dma_buffer);

    (void)memset(&queue, 0, sizeof(queue));
    if ((HAL_DMAEx_List_BuildNode(&config, &node) != HAL_OK) ||
        (HAL_DMAEx_List_InsertNode_Tail(&queue, &node) != HAL_OK) ||
        (HAL_DMAEx_List_SetCircularMode(&queue) != HAL_OK))
    {
        return -1;
    }

    dma.Instance = GPDMA1_Channel7;
    dma.InitLinkedList.Priority = DMA_LOW_PRIORITY_HIGH_WEIGHT;
    dma.InitLinkedList.LinkStepMode = DMA_LSM_FULL_EXECUTION;
    dma.InitLinkedList.LinkAllocatedPort = DMA_LINK_ALLOCATED_PORT0;
    dma.InitLinkedList.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
    dma.InitLinkedList.LinkedListMode = DMA_LINKEDLIST_CIRCULAR;
    if ((HAL_DMAEx_List_Init(&dma) != HAL_OK) || (HAL_DMAEx_List_LinkQ(&dma, &queue) != HAL_OK))
    {
        return -2;
    }
    __HAL_RCC_RIFSC_CLK_ENABLE();
    if (HAL_DMA_ConfigChannelAttributes(&dma, DMA_CHANNEL_SEC | DMA_CHANNEL_PRIV |
                                                  DMA_CHANNEL_SRC_SEC | DMA_CHANNEL_DEST_SEC) != HAL_OK)
    {
        return -3;
    }
    HAL_NVIC_SetPriority(GPDMA1_Channel7_IRQn, XL_AUDIO_IRQ_PRIORITY, 0U);
    HAL_NVIC_EnableIRQ(GPDMA1_Channel7_IRQn);
    return 0;
}

/** @brief SAI2 block B: I2S master transmitter, 48 kHz, 16-bit, two slots. */
static int init_sai(void)
{
    (void)memset(&sai, 0, sizeof(sai));
    sai.Instance = SAI2_Block_B;
    sai.Init.AudioMode = SAI_MODEMASTER_TX;
    sai.Init.Synchro = SAI_ASYNCHRONOUS;
    sai.Init.OutputDrive = SAI_OUTPUTDRIVE_ENABLE;
    sai.Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
    sai.Init.FIFOThreshold = SAI_FIFOTHRESHOLD_HF;
    sai.Init.AudioFrequency = SAI_AUDIO_FREQUENCY_48K;
    sai.Init.SynchroExt = SAI_SYNCEXT_DISABLE;
    sai.Init.MckOutput = SAI_MCK_OUTPUT_DISABLE;
    sai.Init.MckOverSampling = SAI_MCK_OVERSAMPLING_DISABLE;
    sai.Init.MonoStereoMode = SAI_STEREOMODE;
    sai.Init.CompandingMode = SAI_NOCOMPANDING;
    sai.Init.TriState = SAI_OUTPUT_NOTRELEASED;
    if (HAL_SAI_InitProtocol(&sai, SAI_I2S_STANDARD, SAI_PROTOCOL_DATASIZE_16BIT, 2U) != HAL_OK)
    {
        return -1;
    }
    __HAL_LINKDMA(&sai, hdmatx, dma);
    return 0;
}

/*******************************************************************************/
/*                         INTERRUPTS AND HAL CALLBACKS                        */
/*******************************************************************************/

/** @brief GPDMA1 channel 7: the speaker stream (overrides the weak vector). */
void GPDMA1_Channel7_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&dma);
}

/** @brief The first half was sent: refill it. */
void HAL_SAI_TxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
    if (hsai == &sai)
    {
        fill_half(&dma_buffer[0]);
    }
}

/** @brief The second half was sent: refill it. */
void HAL_SAI_TxCpltCallback(SAI_HandleTypeDef *hsai)
{
    if (hsai == &sai)
    {
        fill_half(&dma_buffer[2U * XL_AUDIO_HALF_FRAMES]);
    }
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

int xl_audio_out_init(void)
{
    uint32_t index;

    if (initialized)
    {
        return 0;
    }
    for (index = 0U; index < XL_AUDIO_SINE_ENTRIES; ++index)
    {
        sine[index] = (int16_t)lrintf(32767.0f * sinf((6.2831853f * (float)index) / (float)XL_AUDIO_SINE_ENTRIES));
    }
    ring_head = 0U;
    ring_tail = 0U;
    if (init_clock() != 0)
    {
        return -1;
    }
    init_pins();
    if (init_dma() != 0)
    {
        return -2;
    }
    if (init_sai() != 0)
    {
        return -3;
    }
    initialized = 1;
    return 0;
}

int xl_audio_out_start(void)
{
    if (!initialized)
    {
        return -1;
    }
    if (running)
    {
        return 0;
    }
    (void)memset(dma_buffer, 0, sizeof(dma_buffer));
    if (HAL_SAI_Transmit_DMA(&sai, (uint8_t *)dma_buffer, XL_AUDIO_DMA_SAMPLES) != HAL_OK)
    {
        return -2;
    }
    running = 1;
    return 0;
}

void xl_audio_out_stop(void)
{
    if (running)
    {
        (void)HAL_SAI_DMAStop(&sai);
        running = 0;
    }
}

int xl_audio_out_running(void)
{
    return running;
}

uint32_t xl_audio_out_write(const int16_t *samples, uint32_t count)
{
    uint32_t head = ring_head;
    uint32_t written = 0U;

    if (samples == NULL)
    {
        return 0U;
    }
    while (written < count)
    {
        const uint32_t next = ring_next(head);

        if (next == ring_tail)
        {
            break; /* full */
        }
        /* The speaker's power limit (see XL_AUDIO_OUT_CEILING_Q15). */
        ring[head] = (int16_t)(((int32_t)samples[written] * XL_AUDIO_OUT_CEILING_Q15) >> 15);
        ++written;
        head = next;
    }
    __DMB(); /* the samples before the index that publishes them */
    ring_head = head;
    return written;
}

uint32_t xl_audio_out_queued(void)
{
    const uint32_t head = ring_head;
    const uint32_t tail = ring_tail;

    return (head >= tail) ? (head - tail) : ((XL_AUDIO_RING_SAMPLES - tail) + head);
}

void xl_audio_out_flush(void)
{
    const uint32_t primask = __get_PRIMASK();

    __disable_irq();
    ring_tail = ring_head;
    __set_PRIMASK(primask);
}

uint32_t xl_audio_out_tone(uint32_t frequency_hz, uint32_t duration_ms, int16_t amplitude)
{
    const uint32_t total = (XL_AUDIO_RATE_HZ / 1000U) * duration_ms;
    /* Phase step in 1/65536 of a table entry. */
    const uint32_t step = (uint32_t)(((uint64_t)frequency_hz * XL_AUDIO_SINE_ENTRIES * 65536U) / XL_AUDIO_RATE_HZ);
    int16_t block[128];
    uint32_t phase = 0U;
    uint32_t done = 0U;

    while (done < total)
    {
        uint32_t count = total - done;
        uint32_t index;

        if (count > (sizeof(block) / sizeof(block[0])))
        {
            count = sizeof(block) / sizeof(block[0]);
        }
        for (index = 0U; index < count; ++index)
        {
            const uint32_t n = done + index;
            int32_t value = ((int32_t)sine[(phase >> 16) % XL_AUDIO_SINE_ENTRIES] * amplitude) >> 15;

            if (n < XL_AUDIO_FADE_SAMPLES)
            {
                value = (value * (int32_t)n) / (int32_t)XL_AUDIO_FADE_SAMPLES;
            }
            else if ((total - n) < XL_AUDIO_FADE_SAMPLES)
            {
                value = (value * (int32_t)(total - n)) / (int32_t)XL_AUDIO_FADE_SAMPLES;
            }
            block[index] = (int16_t)value;
            phase += step;
        }
        if (xl_audio_out_write(block, count) != count)
        {
            return done; /* ring full */
        }
        done += count;
    }
    return done;
}

uint32_t xl_audio_out_underruns(void)
{
    return underruns;
}
