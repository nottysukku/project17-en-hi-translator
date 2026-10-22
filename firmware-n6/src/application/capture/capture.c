/**
 * @file     capture.c
 * @brief    Audio capture over the console UART; implements capture.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "capture.h"

#include <string.h>

#include "app_threads.h"
#include "drv_uart.h" /* brings the HAL and CMSIS (__DMB) */
#include "tx_api.h"

/** @brief Microphone queue: 2 s of 16 kHz audio, so a stalled link loses nothing short of that. */
#define CAPTURE_AUDIO_SAMPLES 32768U
/** @brief Log text queue. */
#define CAPTURE_TEXT_BYTES 4096U
/** @brief Room a full logger record needs (LOGGER_RECORD_SIZE). */
#define CAPTURE_TEXT_RECORD_BYTES 512U
/** @brief Largest audio payload per packet, in samples (about 22 ms on the wire at 921600 Bd). */
#define CAPTURE_PACKET_SAMPLES 1024U
/** @brief Largest text payload per packet. */
#define CAPTURE_PACKET_TEXT 256U
/** @brief Header (sync, type, seq, length) and CRC around a payload. */
#define CAPTURE_HEADER_BYTES 6U
#define CAPTURE_CRC_BYTES 2U

static int16_t audio_queue[CAPTURE_AUDIO_SAMPLES];
static volatile uint32_t audio_write; /* producer: microphone reader */
static volatile uint32_t audio_read;  /* consumer: sender thread */
static volatile uint32_t audio_dropped; /* producer-owned running total */
static uint32_t audio_dropped_reported;   /* consumer-owned */

static uint8_t text_queue[CAPTURE_TEXT_BYTES];
static volatile uint32_t text_write; /* producer: logger, serialized by its mutex */
static volatile uint32_t text_read;  /* consumer: sender thread */

/** @brief One packet being sent; the UART driver does not copy, so it stays put until sent. */
static uint8_t packet[CAPTURE_HEADER_BYTES + (CAPTURE_PACKET_SAMPLES * 2U) + CAPTURE_CRC_BYTES];
static uint8_t sequence;

static TX_THREAD capture_thread;
static ULONG capture_thread_stack[APP_CAPTURE_THREAD_STACK_BYTES / sizeof(ULONG)];

/** @brief CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). */
static uint16_t crc16(const uint8_t *data, uint32_t length)
{
    uint16_t crc = 0xFFFFU;
    uint32_t i;
    while (length-- > 0U)
    {
        crc ^= (uint16_t)((uint16_t)*data++ << 8);
        for (i = 0U; i < 8U; ++i)
        {
            crc = (crc & 0x8000U) ? (uint16_t)((crc << 1) ^ 0x1021U) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

/** @brief Frame the payload already at packet[6] and start sending it; returns false if busy. */
static bool send_packet(uint8_t type, uint32_t payload_bytes)
{
    uint16_t crc;
    packet[0] = 0xA5U;
    packet[1] = 0x5AU;
    packet[2] = type;
    packet[3] = sequence++;
    packet[4] = (uint8_t)(payload_bytes & 0xFFU);
    packet[5] = (uint8_t)(payload_bytes >> 8);
    crc = crc16(&packet[2], 4U + payload_bytes);
    packet[CAPTURE_HEADER_BYTES + payload_bytes] = (uint8_t)(crc & 0xFFU);
    packet[CAPTURE_HEADER_BYTES + payload_bytes + 1U] = (uint8_t)(crc >> 8);
    return drv_uart_write(packet, CAPTURE_HEADER_BYTES + payload_bytes + CAPTURE_CRC_BYTES);
}

/** @brief Send whatever is queued: drop reports first, then text, then audio. */
static void capture_thread_entry(ULONG argument)
{
    (void)argument;
    for (;;)
    {
        uint32_t dropped;
        uint32_t text_available;
        uint32_t audio_available;

        if (!drv_uart_tx_ready())
        {
            tx_thread_sleep(1U);
            continue;
        }
        dropped = audio_dropped - audio_dropped_reported;
        text_available = text_write - text_read;
        audio_available = audio_write - audio_read;

        if (dropped != 0U)
        {
            audio_dropped_reported += dropped;
            memcpy(&packet[CAPTURE_HEADER_BYTES], &dropped, sizeof(dropped));
            (void)send_packet('D', sizeof(dropped));
        }
        else if (text_available != 0U)
        {
            uint32_t n = (text_available < CAPTURE_PACKET_TEXT) ? text_available : CAPTURE_PACKET_TEXT;
            uint32_t i;
            for (i = 0U; i < n; ++i)
            {
                packet[CAPTURE_HEADER_BYTES + i] = text_queue[(text_read + i) % CAPTURE_TEXT_BYTES];
            }
            __DMB();
            text_read += n;
            (void)send_packet('T', n);
        }
        else if (audio_available != 0U)
        {
            uint32_t n = (audio_available < CAPTURE_PACKET_SAMPLES) ? audio_available : CAPTURE_PACKET_SAMPLES;
            uint32_t i;
            for (i = 0U; i < n; ++i)
            {
                int16_t s = audio_queue[(audio_read + i) % CAPTURE_AUDIO_SAMPLES];
                packet[CAPTURE_HEADER_BYTES + (2U * i)] = (uint8_t)((uint16_t)s & 0xFFU);
                packet[CAPTURE_HEADER_BYTES + (2U * i) + 1U] = (uint8_t)((uint16_t)s >> 8);
            }
            __DMB();
            audio_read += n;
            (void)send_packet('A', 2U * n);
        }
        else
        {
            tx_thread_sleep(1U);
        }
    }
}

uint32_t capture_start(void)
{
    audio_write = audio_read = audio_dropped = audio_dropped_reported = 0U;
    text_write = text_read = 0U;
    return (uint32_t)tx_thread_create(&capture_thread,
                                      "capture_thread",
                                      capture_thread_entry,
                                      0U,
                                      capture_thread_stack,
                                      sizeof(capture_thread_stack),
                                      APP_CAPTURE_THREAD_PRIORITY,
                                      APP_CAPTURE_THREAD_PRIORITY,
                                      TX_NO_TIME_SLICE,
                                      TX_AUTO_START);
}

void capture_push_audio(const int16_t *pcm, uint32_t samples)
{
    uint32_t free_samples = CAPTURE_AUDIO_SAMPLES - (audio_write - audio_read);
    uint32_t n = (samples < free_samples) ? samples : free_samples;
    uint32_t i;

    for (i = 0U; i < n; ++i)
    {
        audio_queue[(audio_write + i) % CAPTURE_AUDIO_SAMPLES] = pcm[i];
    }
    __DMB();
    audio_write += n;
    if (n < samples)
    {
        audio_dropped += samples - n;
    }
}

bool capture_log_try_write(const uint8_t *data, size_t length)
{
    uint32_t free_bytes = CAPTURE_TEXT_BYTES - (text_write - text_read);
    uint32_t i;

    if (length > free_bytes)
    {
        return false;
    }
    for (i = 0U; i < length; ++i)
    {
        text_queue[(text_write + i) % CAPTURE_TEXT_BYTES] = data[i];
    }
    __DMB();
    text_write += (uint32_t)length;
    return true;
}

bool capture_log_ready(void)
{
    return (CAPTURE_TEXT_BYTES - (text_write - text_read)) >= CAPTURE_TEXT_RECORD_BYTES;
}
