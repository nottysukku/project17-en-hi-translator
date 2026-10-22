/**
 * @file     app_threads.h
 * @brief    Every thread of the firmware in one table: priority and stack.
 * @details  A lower number preempts a higher one.
 *
 *           | Prio | Thread   | Code                  | Job                                |
 *           |------|----------|-----------------------|------------------------------------|
 *           |    5 | startup  | app_startup.c         | starts the modules, then exits     |
 *           |    8 | asr_mel  | asr_engine/mel/        | audio to log-mel                   |
 *           |    9 | asr_norm | asr_engine/norm/       | normalizes the feature window      |
 *           |   10 | led      | application/led/      | heartbeat LED                      |
 *           |   11 | asr_net  | asr_engine/net/        | NPU inference + CTC                |
 *           |   12 | display  | application/display/  | redraws the transcript             |
 *
 *           asr_mel must keep pace with capture; display is lowest, a late
 *           frame costs nothing. asr_engine reads the @c ASR_CONF_* values
 *           through asr_engine_conf.h.
 * @author   Nikita Prismakov
 *
 * SPDX-FileCopyrightText: 2026 Nikita Prismakov
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_CONFIG_APP_THREADS_H
#define APPLICATION_CONFIG_APP_THREADS_H

/*******************************************************************************/
/*                                PUBLIC DEFINES                               */
/*******************************************************************************/

/* ThreadX: a lower number is a higher priority. The list is in priority order,
   so the audio path (mel, norm) preempts everything but the startup thread. */

/** @brief Startup thread (app_startup.c): brings the modules up, then exits. */
#define APP_STARTUP_THREAD_PRIORITY       5U
#define APP_STARTUP_THREAD_STACK_BYTES    2048U

/** @brief asr_engine mel stage: microphone audio to log-mel features. */
#define ASR_CONF_MEL_THREAD_PRIORITY      8U
#define ASR_CONF_MEL_THREAD_STACK_BYTES   4096U

/** @brief asr_engine normalization stage. */
#define ASR_CONF_NORM_THREAD_PRIORITY     9U
#define ASR_CONF_NORM_THREAD_STACK_BYTES  4096U

/** @brief Heartbeat LED (led/led_thread.c). */
#define APP_LED_THREAD_PRIORITY           10U
#define APP_LED_THREAD_STACK_BYTES        1024U

/** @brief Audio capture sender (capture/capture.c), data-collection builds only. */
#define APP_CAPTURE_THREAD_PRIORITY       10U
#define APP_CAPTURE_THREAD_STACK_BYTES    2048U

/** @brief asr_engine network stage: NPU inference and CTC decoding. */
#define ASR_CONF_NET_THREAD_PRIORITY      11U
#define ASR_CONF_NET_THREAD_STACK_BYTES   4096U

/** @brief Transcript on the LCD (display/display_thread.c). */
#define APP_DISPLAY_THREAD_PRIORITY       12U
#if defined(APP_EDU_MODE)
/* EDU mode: the display thread also checks the answers (edu_math.c) and reads the
   touch screen; its deepest call chain is measured by edu/tools/stack_report.py. */
#define APP_DISPLAY_THREAD_STACK_BYTES    4096U
#elif defined(APP_TRANSLATOR_MODE)
/* Translator mode: the display thread draws the screen, reads the touch screen and runs the
   speaker router; measured with -fstack-usage like EDU mode, with margin. */
#define APP_DISPLAY_THREAD_STACK_BYTES    6144U
#else
#define APP_DISPLAY_THREAD_STACK_BYTES    2048U
#endif

#endif /* APPLICATION_CONFIG_APP_THREADS_H */
