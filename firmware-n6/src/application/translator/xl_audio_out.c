/**
 * @file     xl_audio_out.c
 * @brief    Translator audio on the 3.5 mm jack; implements xl_audio_out.h.
 * @details  Out: GPDMA1 channel 7 runs one circular node over a stereo double buffer into SAI1
 *           block A (I2S master, MCLK = 256 fs) and the WM8904. The SAI half- and full-transfer
 *           callbacks refill the half just sent from the speech ring or the music ring.
 *           In: GPDMA1 channel 5 runs one circular node from SAI2 block B (I2S slave receiver,
 *           clocked by the WBA) into a stereo double buffer; its callbacks mix each half to mono
 *           into the music ring.
 *           The DMA buffers and nodes are non-cacheable, as the microphone's capture area
 *           (drv_pdm_dma.c), so no cache maintenance is needed; the rings are only touched by the
 *           CPU. Both DMA interrupts have the same priority, so they never interrupt each other:
 *           each index has one writer.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "xl_audio_out.h"

#include <math.h>
#include <string.h>

#include "logger.h"
#include "mcu_clock.h"
#include "stm32n6570_discovery_bus.h"
#include "stm32n6xx_hal.h"
#include "stm32n6xx_ll_rcc.h"
#include "wm8904.h"

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief Stereo frames per DMA half (5.3 ms): latency and interrupt rate, both directions. */
#define XL_AUDIO_HALF_FRAMES 256U

/** @brief Samples (both slots) of a whole double buffer. */
#define XL_AUDIO_DMA_SAMPLES (2U * XL_AUDIO_HALF_FRAMES * 2U)

/** @brief Mono speech samples the ring holds (4 s); one slot stays empty to tell full from empty. */
#define XL_AUDIO_RING_SAMPLES (4U * XL_AUDIO_RATE_HZ)

/** @brief NVIC priority of the DMA and SAI interrupts; they call no RTOS service. */
#define XL_AUDIO_IRQ_PRIORITY 6U

/** @brief Entries of the sine table used by xl_audio_out_tone(). */
#define XL_AUDIO_SINE_ENTRIES 256U

/** @brief Fade-in and fade-out of a tone, in samples (5 ms). */
#define XL_AUDIO_FADE_SAMPLES (XL_AUDIO_RATE_HZ / 200U)

/** @brief Gain change per sample when the source changes: 0 to full in 240 samples (5 ms). */
#define XL_AUDIO_SWITCH_STEP (32768U / 240U)

/** @brief WM8904 on I2C2: 0x1A, as the 8-bit address the BSP bus functions take. */
#define XL_CODEC_I2C_ADDRESS 0x34U

/** @brief Music ring: mono samples, a power of two (170 ms at 48 kHz). */
#define XL_MUSIC_RING 8192U
#define XL_MUSIC_MASK (XL_MUSIC_RING - 1U)

/** @brief Music samples kept queued (43 ms at 48 kHz): the latency, and room for clock drift. */
#define XL_MUSIC_TARGET 2048U

/** @brief Below this many received frames per second there is no music clock. */
#define XL_MUSIC_MIN_RATE 4000U

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

static SAI_HandleTypeDef sai_out;  /* SAI1 block A -> WM8904 */
static SAI_HandleTypeDef sai_in;   /* SAI2 block B <- WBA */
static DMA_HandleTypeDef dma_out;  /* GPDMA1 channel 7 */
static DMA_HandleTypeDef dma_in;   /* GPDMA1 channel 5 */
static DMA_QListTypeDef queue_out;
static DMA_QListTypeDef queue_in;
static DMA_NodeTypeDef node_out __NON_CACHEABLE __attribute__((aligned(32)));
static DMA_NodeTypeDef node_in __NON_CACHEABLE __attribute__((aligned(32)));

/** @brief Stereo double buffers (L, R, L, R, ...). */
static int16_t out_buffer[XL_AUDIO_DMA_SAMPLES] __NON_CACHEABLE __attribute__((aligned(32)));
static int16_t in_buffer[XL_AUDIO_DMA_SAMPLES] __NON_CACHEABLE __attribute__((aligned(32)));

/** @brief Speech waiting to be played; written by the application, read by the out interrupt. */
static int16_t ring[XL_AUDIO_RING_SAMPLES] __attribute__((section(".psram_bss"), aligned(32)));
static volatile uint32_t ring_head; /* written by xl_audio_out_write() only */
static volatile uint32_t ring_tail; /* written by the out interrupt (flush masks interrupts) */
static volatile uint32_t underruns;

/** @brief Music received from the WBA, mono; head written by the in interrupt, tail by the out one. */
static int16_t music[XL_MUSIC_RING];
static volatile uint32_t music_head;
static volatile uint32_t music_tail;
static volatile uint32_t music_frames;    /* frames received since start-up */
static volatile uint32_t music_overruns;
static volatile int music_error;          /* the receiver stopped on a frame-sync error */

/** @brief Reading the music: fractional position, step (input samples per output sample), control. */
static float music_frac;
static float music_step = 1.0f;
static volatile float music_nominal = 1.0f; /* WBA rate / jack rate, from xl_audio_out_poll() */
static float music_integral;
static float music_level;
static int music_priming = 1;

/** @brief The source choice: wanted (thread), playing and the fade (out interrupt). */
static volatile xl_audio_source_t wanted = XL_AUDIO_SPEECH;
static volatile xl_audio_source_t playing = XL_AUDIO_SPEECH;
static uint32_t fade_gain = 32768U;
static int fade_dir;

/** @brief The jack's real sample rate (47991 Hz from the 49.14 MHz audio clock). */
static float jack_rate = 48000.0f;

/** @brief Measured WBA rate and the measurement window. */
static uint32_t music_rate;
static uint32_t rate_frames;
static uint32_t rate_since;

static WM8904_Object_t codec;
static volatile int running;
static int initialized;
static int16_t sine[XL_AUDIO_SINE_ENTRIES];

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

static uint32_t ring_next(uint32_t index)
{
    return ((index + 1U) == XL_AUDIO_RING_SAMPLES) ? 0U : (index + 1U);
}

/** @brief Start reading the music afresh: wait for XL_MUSIC_TARGET samples, no correction yet. */
static void music_restart(void)
{
    music_priming = 1;
    music_frac = 0.0f;
    music_integral = 0.0f;
    music_level = (float)XL_MUSIC_TARGET;
    music_step = music_nominal;
}

/** @brief The next music sample at the jack's rate: linear interpolation between ring samples. */
static int32_t next_music(void)
{
    uint32_t tail = music_tail;
    uint32_t avail = (music_head - tail) & XL_MUSIC_MASK;
    int32_t a;
    int32_t b;
    int32_t value;

    if (music_priming)
    {
        if (avail < XL_MUSIC_TARGET)
        {
            return 0;
        }
        music_priming = 0;
        if (fade_dir == 0)
        {
            fade_gain = 0U; /* the music starts now, mid-waveform: fade it in */
            fade_dir = 1;
        }
    }
    if (avail < 2U)
    {
        music_restart(); /* the WBA stopped or fell behind: refill before playing on */
        return 0;
    }
    a = music[tail];
    b = music[(tail + 1U) & XL_MUSIC_MASK];
    value = a + (int32_t)lrintf((float)(b - a) * music_frac);
    music_frac += music_step;
    while ((music_frac >= 1.0f) && (avail >= 2U))
    {
        tail = (tail + 1U) & XL_MUSIC_MASK;
        avail--;
        music_frac -= 1.0f;
    }
    music_tail = tail;
    return value;
}

/**
 * @brief After each half: read faster when the music ring fills, slower when it empties.
 * @note A PI controller on the smoothed level: the two boards' clocks differ by up to about 1 %
 *       (the N6's audio PLL runs from its internal RC oscillator), which the integral takes up;
 *       the proportional part settles the level in a few seconds without audible pitch change.
 */
static void music_control(void)
{
    const uint32_t fill = (music_head - music_tail) & XL_MUSIC_MASK;
    float error;
    float step;

    if (music_priming)
    {
        return;
    }
    music_level += ((float)fill - music_level) * (1.0f / 32.0f);
    error = (music_level - (float)XL_MUSIC_TARGET) / (float)XL_MUSIC_TARGET;
    music_integral += error * 1.6e-5f;
    if (music_integral > 0.03f)
    {
        music_integral = 0.03f;
    }
    else if (music_integral < -0.03f)
    {
        music_integral = -0.03f;
    }
    step = music_nominal * (1.0f + (0.02f * error) + music_integral);
    if (step > (music_nominal * 1.05f))
    {
        step = music_nominal * 1.05f;
    }
    else if (step < (music_nominal * 0.95f))
    {
        step = music_nominal * 0.95f;
    }
    music_step = step;
}

/**
 * @brief Refill one output half; runs in the out DMA interrupt.
 * @param[out] out First stereo sample of the half.
 */
static void fill_half(int16_t *out)
{
    uint32_t tail = ring_tail;
    const uint32_t head = ring_head;
    uint32_t frame;
    int speech_given = 0;
    int speech_short = 0;

    if ((wanted != playing) && (fade_dir == 0))
    {
        fade_dir = -1;
    }
    if (playing != XL_AUDIO_MUSIC)
    {
        music_tail = music_head; /* the music is not heard: keep its ring empty */
    }
    for (frame = 0U; frame < XL_AUDIO_HALF_FRAMES; ++frame)
    {
        int32_t sample = 0;

        if (playing == XL_AUDIO_MUSIC)
        {
            sample = next_music();
        }
        else if (tail != head)
        {
            sample = ring[tail];
            tail = ring_next(tail);
            speech_given = 1;
        }
        else if (speech_given)
        {
            speech_short = 1; /* the ring ran dry in the middle of this half */
        }
        sample = (sample * (int32_t)fade_gain) >> 15;
        if (fade_dir < 0)
        {
            fade_gain = (fade_gain > XL_AUDIO_SWITCH_STEP) ? (fade_gain - XL_AUDIO_SWITCH_STEP) : 0U;
            if (fade_gain == 0U)
            {
                if (wanted != playing)
                {
                    if (playing == XL_AUDIO_SPEECH)
                    {
                        tail = head; /* music always wins: the rest of the speech is dropped */
                    }
                    playing = wanted;
                    music_restart();
                }
                fade_dir = 1;
            }
        }
        else if (fade_dir > 0)
        {
            fade_gain += XL_AUDIO_SWITCH_STEP;
            if (fade_gain >= 32768U)
            {
                fade_gain = 32768U;
                fade_dir = 0;
            }
        }
        out[2U * frame] = (int16_t)sample;
        out[(2U * frame) + 1U] = (int16_t)sample;
    }
    ring_tail = tail;
    if (speech_short)
    {
        ++underruns;
    }
    if (playing == XL_AUDIO_MUSIC)
    {
        music_control();
    }
}

/**
 * @brief Mix one received half to mono into the music ring; runs in the in DMA interrupt.
 * @param[in] in First stereo sample of the half (left slot first).
 */
static void take_half(const int16_t *in)
{
    uint32_t head = music_head;
    const uint32_t tail = music_tail;
    uint32_t frame;

    for (frame = 0U; frame < XL_AUDIO_HALF_FRAMES; ++frame)
    {
        const uint32_t next = (head + 1U) & XL_MUSIC_MASK;

        if (next == tail)
        {
            ++music_overruns;
            break;
        }
        music[head] = (int16_t)(((int32_t)in[2U * frame] + (int32_t)in[(2U * frame) + 1U]) / 2);
        head = next;
    }
    music_head = head;
    music_frames += XL_AUDIO_HALF_FRAMES;
}

/**
 * @brief Make sure IC8 runs from PLL3 and select it as the kernel clock of SAI1 and SAI2.
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
    __HAL_RCC_SAI1_CONFIG(RCC_SAI1CLKSOURCE_IC8);
    __HAL_RCC_SAI2_CONFIG(RCC_SAI2CLKSOURCE_IC8);
    __HAL_RCC_SAI1_CLK_ENABLE();
    __HAL_RCC_SAI2_CLK_ENABLE();
    return 0;
}

static void init_pin(GPIO_TypeDef *port, uint32_t pin, uint32_t pull, uint8_t alternate)
{
    GPIO_InitTypeDef gpio = {0};

    gpio.Pin = pin;
    gpio.Mode = GPIO_MODE_AF_PP;
    gpio.Pull = pull;
    gpio.Speed = GPIO_SPEED_FREQ_HIGH;
    gpio.Alternate = alternate;
    HAL_GPIO_Init(port, &gpio);
    HAL_GPIO_ConfigPinAttributes(port, pin, GPIO_PIN_SEC | GPIO_PIN_NPRIV);
}

/**
 * @brief SAI1_A to the codec (PG7 MCLK, PB6 SCK, PB7 SD, PB0 FS, AF6, the DK's own wiring) and
 *        SAI2_B from the WBA (PE12 SCK, PE13 FS, PE7 SD, AF10, inputs with pull-downs so they
 *        rest low while the TMUX1574 is open).
 */
static void init_pins(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    init_pin(GPIOG, GPIO_PIN_7, GPIO_NOPULL, GPIO_AF6_SAI1);
    init_pin(GPIOB, GPIO_PIN_6, GPIO_NOPULL, GPIO_AF6_SAI1);
    init_pin(GPIOB, GPIO_PIN_7, GPIO_NOPULL, GPIO_AF6_SAI1);
    init_pin(GPIOB, GPIO_PIN_0, GPIO_NOPULL, GPIO_AF6_SAI1);
    init_pin(GPIOE, GPIO_PIN_12, GPIO_PULLDOWN, GPIO_AF10_SAI2);
    init_pin(GPIOE, GPIO_PIN_13, GPIO_PULLDOWN, GPIO_AF10_SAI2);
    init_pin(GPIOE, GPIO_PIN_7, GPIO_PULLDOWN, GPIO_AF10_SAI2);
}

/**
 * @brief One circular node over a double buffer, on a GPDMA1 channel.
 * @note Port allocation as the microphone's channel: memory on port 1, peripheral on port 0.
 */
static int init_dma(DMA_HandleTypeDef *dma, DMA_QListTypeDef *queue, DMA_NodeTypeDef *node,
                    DMA_Channel_TypeDef *channel, uint32_t request, int to_peripheral, uint32_t peripheral,
                    int16_t *buffer, IRQn_Type irq)
{
    DMA_NodeConfTypeDef config = {0};

    __HAL_RCC_GPDMA1_CLK_ENABLE();
    config.NodeType = DMA_GPDMA_LINEAR_NODE;
    config.Init.Request = request;
    config.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
    config.Init.Direction = to_peripheral ? DMA_MEMORY_TO_PERIPH : DMA_PERIPH_TO_MEMORY;
    config.Init.SrcInc = to_peripheral ? DMA_SINC_INCREMENTED : DMA_SINC_FIXED;
    config.Init.DestInc = to_peripheral ? DMA_DINC_FIXED : DMA_DINC_INCREMENTED;
    config.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_HALFWORD;
    config.Init.DestDataWidth = DMA_DEST_DATAWIDTH_HALFWORD;
    config.Init.SrcBurstLength = 1U;
    config.Init.DestBurstLength = 1U;
    config.Init.TransferAllocatedPort = to_peripheral ? (DMA_SRC_ALLOCATED_PORT1 | DMA_DEST_ALLOCATED_PORT0)
                                                      : (DMA_SRC_ALLOCATED_PORT0 | DMA_DEST_ALLOCATED_PORT1);
    config.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
    config.Init.Mode = DMA_NORMAL;
    config.TriggerConfig.TriggerPolarity = DMA_TRIG_POLARITY_MASKED;
    config.DataHandlingConfig.DataExchange = DMA_EXCHANGE_NONE;
    config.DataHandlingConfig.DataAlignment = DMA_DATA_RIGHTALIGN_ZEROPADDED;
    config.SrcSecure = DMA_CHANNEL_SRC_SEC;
    config.DestSecure = DMA_CHANNEL_DEST_SEC;
    config.SrcAddress = to_peripheral ? (uint32_t)buffer : peripheral;
    config.DstAddress = to_peripheral ? peripheral : (uint32_t)buffer;
    config.DataSize = XL_AUDIO_DMA_SAMPLES * sizeof(int16_t);

    (void)memset(queue, 0, sizeof(*queue));
    if ((HAL_DMAEx_List_BuildNode(&config, node) != HAL_OK) || (HAL_DMAEx_List_InsertNode_Tail(queue, node) != HAL_OK) ||
        (HAL_DMAEx_List_SetCircularMode(queue) != HAL_OK))
    {
        return -1;
    }
    dma->Instance = channel;
    dma->InitLinkedList.Priority = DMA_LOW_PRIORITY_HIGH_WEIGHT;
    dma->InitLinkedList.LinkStepMode = DMA_LSM_FULL_EXECUTION;
    dma->InitLinkedList.LinkAllocatedPort = DMA_LINK_ALLOCATED_PORT0;
    dma->InitLinkedList.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
    dma->InitLinkedList.LinkedListMode = DMA_LINKEDLIST_CIRCULAR;
    if ((HAL_DMAEx_List_Init(dma) != HAL_OK) || (HAL_DMAEx_List_LinkQ(dma, queue) != HAL_OK))
    {
        return -2;
    }
    __HAL_RCC_RIFSC_CLK_ENABLE();
    if (HAL_DMA_ConfigChannelAttributes(dma, DMA_CHANNEL_SEC | DMA_CHANNEL_PRIV | DMA_CHANNEL_SRC_SEC |
                                                 DMA_CHANNEL_DEST_SEC) != HAL_OK)
    {
        return -3;
    }
    HAL_NVIC_SetPriority(irq, XL_AUDIO_IRQ_PRIORITY, 0U);
    HAL_NVIC_EnableIRQ(irq);
    return 0;
}

/** @brief I2S, 16-bit, two slots in a 32-bit frame: the WBA's framing and the WM8904's. */
static int init_sai(SAI_HandleTypeDef *sai, SAI_Block_TypeDef *block, uint32_t mode, uint32_t mclk,
                    DMA_HandleTypeDef *dma)
{
    (void)memset(sai, 0, sizeof(*sai));
    sai->Instance = block;
    sai->Init.AudioMode = mode;
    sai->Init.Synchro = SAI_ASYNCHRONOUS;
    sai->Init.OutputDrive = SAI_OUTPUTDRIVE_ENABLE;
    sai->Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
    sai->Init.FIFOThreshold = SAI_FIFOTHRESHOLD_HF;
    sai->Init.AudioFrequency = SAI_AUDIO_FREQUENCY_48K;
    sai->Init.SynchroExt = SAI_SYNCEXT_DISABLE;
    sai->Init.MckOutput = mclk;
    sai->Init.MckOverSampling = SAI_MCK_OVERSAMPLING_DISABLE;
    sai->Init.MonoStereoMode = SAI_STEREOMODE;
    sai->Init.CompandingMode = SAI_NOCOMPANDING;
    sai->Init.TriState = SAI_OUTPUT_NOTRELEASED;
    if (HAL_SAI_InitProtocol(sai, SAI_I2S_STANDARD, SAI_PROTOCOL_DATASIZE_16BIT, 2U) != HAL_OK)
    {
        return -1;
    }
    if (mode == SAI_MODEMASTER_TX)
    {
        __HAL_LINKDMA(sai, hdmatx, *dma);
    }
    else
    {
        __HAL_LINKDMA(sai, hdmarx, *dma);
    }
    return 0;
}

/** @brief The jack's sample rate from the SAI1 kernel clock and the divider HAL chose. */
static float measure_jack_rate(void)
{
    const uint32_t kernel = HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_SAI1);
    uint32_t mckdiv = (SAI1_Block_A->CR1 & SAI_xCR1_MCKDIV) >> SAI_xCR1_MCKDIV_Pos;

    if (mckdiv == 0U)
    {
        mckdiv = 1U;
    }
    return (kernel == 0U) ? 48000.0f : ((float)kernel / (float)mckdiv / 256.0f);
}

/**
 * @brief The WM8904: found on I2C2, set for the headphone output (the jack) at 48 kHz.
 * @return 0, -4 not found, -5 not configured.
 * @note MCLK must already run (SAI1 is initialized with its master clock on).
 */
static int init_codec(void)
{
    WM8904_IO_t io = {0};
    WM8904_Init_t setup = {0};
    uint32_t id = 0U;
    int32_t volume = (44 + XL_AUDIO_OUT_VOLUME_DB) * 2; /* the driver's 0-100 scale: dB = v / 2 - 44 */

    io.Init = BSP_I2C2_Init;
    io.DeInit = BSP_I2C2_DeInit;
    io.Address = XL_CODEC_I2C_ADDRESS;
    io.WriteReg = BSP_I2C2_WriteReg;
    io.ReadReg = BSP_I2C2_ReadReg;
    io.GetTick = BSP_GetTick;
    if ((WM8904_RegisterBusIO(&codec, &io) != WM8904_OK) || (WM8904_ReadID(&codec, &id) != WM8904_OK) ||
        ((id & WM8904_ID_MASK) != WM8904_ID))
    {
        LOG_ERROR("Audio: no WM8904 on I2C2 (read 0x%04lx); an older DK with a CS42L51 is not supported",
                  (unsigned long)id);
        return -4;
    }
    if (volume < 0)
    {
        volume = 0;
    }
    else if (volume > 99)
    {
        volume = 99;
    }
    setup.InputDevice = WM8904_IN_NONE;
    setup.OutputDevice = WM8904_OUT_HEADPHONE;
    setup.Frequency = WM8904_FREQUENCY_48K;
    setup.Resolution = WM8904_RESOLUTION_16B;
    setup.Volume = (uint32_t)volume;
    if (WM8904_Init(&codec, &setup) != WM8904_OK)
    {
        return -5;
    }
    return 0;
}

/*******************************************************************************/
/*                         INTERRUPTS AND HAL CALLBACKS                        */
/*******************************************************************************/

/** @brief GPDMA1 channel 7: the jack's stream (overrides the weak vector). */
void GPDMA1_Channel7_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&dma_out);
}

/** @brief GPDMA1 channel 5: the music from the WBA. */
void GPDMA1_Channel5_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&dma_in);
}

/** @brief SAI2 block B: the receiver's errors (frame sync lost when the WBA stops). */
void SAI2_B_IRQHandler(void)
{
    HAL_SAI_IRQHandler(&sai_in);
}

void HAL_SAI_TxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
    if (hsai == &sai_out)
    {
        fill_half(&out_buffer[0]);
    }
}

void HAL_SAI_TxCpltCallback(SAI_HandleTypeDef *hsai)
{
    if (hsai == &sai_out)
    {
        fill_half(&out_buffer[2U * XL_AUDIO_HALF_FRAMES]);
    }
}

void HAL_SAI_RxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
    if (hsai == &sai_in)
    {
        take_half(&in_buffer[0]);
    }
}

void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai)
{
    if (hsai == &sai_in)
    {
        take_half(&in_buffer[2U * XL_AUDIO_HALF_FRAMES]);
    }
}

void HAL_SAI_ErrorCallback(SAI_HandleTypeDef *hsai)
{
    if ((hsai == &sai_in) && ((hsai->ErrorCode & (HAL_SAI_ERROR_AFSDET | HAL_SAI_ERROR_LFSDET)) != 0U))
    {
        music_error = 1; /* the DMA was aborted: xl_audio_out_poll() restarts the receiver */
    }
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

int xl_audio_out_init(void)
{
    uint32_t index;
    int status;

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
    music_head = 0U;
    music_tail = 0U;
    if (init_clock() != 0)
    {
        return -1;
    }
    init_pins();
    if ((init_dma(&dma_out, &queue_out, &node_out, GPDMA1_Channel7, GPDMA1_REQUEST_SAI1_A, 1,
                  (uint32_t)&SAI1_Block_A->DR, out_buffer, GPDMA1_Channel7_IRQn) != 0) ||
        (init_dma(&dma_in, &queue_in, &node_in, GPDMA1_Channel5, GPDMA1_REQUEST_SAI2_B, 0,
                  (uint32_t)&SAI2_Block_B->DR, in_buffer, GPDMA1_Channel5_IRQn) != 0))
    {
        return -2;
    }
    /* SAI1 first: its master clock must run before the codec is set up. */
    if ((init_sai(&sai_out, SAI1_Block_A, SAI_MODEMASTER_TX, SAI_MCK_OUTPUT_ENABLE, &dma_out) != 0) ||
        (init_sai(&sai_in, SAI2_Block_B, SAI_MODESLAVE_RX, SAI_MCK_OUTPUT_DISABLE, &dma_in) != 0))
    {
        return -3;
    }
    HAL_NVIC_SetPriority(SAI2_B_IRQn, XL_AUDIO_IRQ_PRIORITY, 0U);
    HAL_NVIC_EnableIRQ(SAI2_B_IRQn);
    jack_rate = measure_jack_rate();
    music_nominal = 48000.0f / jack_rate;
    status = init_codec();
    if (status != 0)
    {
        return status;
    }
    LOG_INFO("Audio: WM8904 found, jack at %lu Hz, headphone volume %d dB", (unsigned long)lrintf(jack_rate),
             (int)XL_AUDIO_OUT_VOLUME_DB);
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
    (void)memset(out_buffer, 0, sizeof(out_buffer));
    if (HAL_SAI_Transmit_DMA(&sai_out, (uint8_t *)out_buffer, XL_AUDIO_DMA_SAMPLES) != HAL_OK)
    {
        return -2;
    }
    if (WM8904_Play(&codec) != WM8904_OK) /* the DAC was muted by its init */
    {
        return -3;
    }
    music_error = 0;
    if (HAL_SAI_Receive_DMA(&sai_in, (uint8_t *)in_buffer, XL_AUDIO_DMA_SAMPLES) != HAL_OK)
    {
        LOG_WARNING("Audio: music input did not start");
    }
    running = 1;
    return 0;
}

void xl_audio_out_stop(void)
{
    if (running)
    {
        (void)HAL_SAI_DMAStop(&sai_in);
        (void)HAL_SAI_DMAStop(&sai_out);
        running = 0;
    }
}

int xl_audio_out_running(void)
{
    return running;
}

void xl_audio_out_select(xl_audio_source_t source)
{
    wanted = source;
    if (!running)
    {
        playing = source; /* no stream, nothing to fade */
    }
}

xl_audio_source_t xl_audio_out_selected(void)
{
    return playing;
}

void xl_audio_out_poll(uint32_t now_ms)
{
    static const uint32_t rates[] = {8000U, 11025U, 16000U, 22050U, 24000U, 32000U, 44100U, 48000U};
    uint32_t elapsed;

    if (!running)
    {
        return;
    }
    if (music_error || (HAL_SAI_GetState(&sai_in) == HAL_SAI_STATE_READY))
    {
        music_error = 0;
        (void)HAL_SAI_Abort(&sai_in);
        (void)HAL_SAI_Receive_DMA(&sai_in, (uint8_t *)in_buffer, XL_AUDIO_DMA_SAMPLES);
    }
    elapsed = now_ms - rate_since;
    if (elapsed >= 1000U)
    {
        const uint32_t frames = music_frames;
        const uint32_t measured = (uint32_t)(((uint64_t)(frames - rate_frames) * 1000U) / elapsed);
        uint32_t best = 0U;
        uint32_t index;

        for (index = 0U; index < (sizeof(rates) / sizeof(rates[0])); ++index)
        {
            const uint32_t diff = (measured > rates[index]) ? (measured - rates[index]) : (rates[index] - measured);
            if ((diff * 50U) <= rates[index]) /* within 2 % */
            {
                best = rates[index];
            }
        }
        if ((measured >= XL_MUSIC_MIN_RATE) && (best == 0U))
        {
            best = measured; /* an unusual rate: take it as measured */
        }
        if (best != music_rate)
        {
            if (best != 0U)
            {
                LOG_INFO("Audio: music input at %lu Hz", (unsigned long)best);
                music_nominal = (float)best / jack_rate;
            }
            music_rate = best;
        }
        rate_frames = frames;
        rate_since = now_ms;
    }
}

uint32_t xl_audio_out_music_rate(void)
{
    return music_rate;
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
