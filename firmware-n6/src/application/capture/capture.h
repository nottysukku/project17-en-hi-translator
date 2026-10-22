/**
 * @file     capture.h
 * @brief    Audio capture over the console UART, for collecting training data with the board mic.
 * @details  Built only with -DAPP_AUDIO_CAPTURE=ON. The microphone samples the recognizer reads are
 *           copied here and sent, together with the log text, as framed packets on the console
 *           UART (921600 8N1). Packet: 0xA5 0x5A type seq len_lo len_hi payload crc_lo crc_hi,
 *           CRC-16/CCITT-FALSE over type..payload. Types: 'A' PCM16 LE mono 16 kHz,
 *           'T' log text, 'D' uint32 LE count of samples dropped because the link fell behind.
 *           cynipsnet_re/datacollect/board_record.py receives them.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_CAPTURE_CAPTURE_H
#define APPLICATION_CAPTURE_CAPTURE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create the sender thread; call before logger_init() and before the microphone starts.
 * @return 0 on success, a ThreadX status otherwise.
 */
uint32_t capture_start(void);

/**
 * @brief Queue microphone samples for sending (single producer: the thread reading the mic).
 * @param[in] pcm PCM16 samples at 16 kHz.
 * @param[in] samples Number of samples.
 */
void capture_push_audio(const int16_t *pcm, uint32_t samples);

/** @brief logger_try_write_t: copy one log record into the text queue; false if it does not fit. */
bool capture_log_try_write(const uint8_t *data, size_t length);

/** @brief logger_output_ready_t: true when the text queue can take a full log record. */
bool capture_log_ready(void);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_CAPTURE_CAPTURE_H */
