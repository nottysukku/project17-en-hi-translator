/**
 * @file     net_cfg.h
 * @brief    Window geometry of the network stage.
 * @author   Nikita Prismakov
 *
 * SPDX-FileCopyrightText: 2026 Nikita Prismakov
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_ENGINE_NET_NET_CFG_H
#define ASR_ENGINE_NET_NET_CFG_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

/* Model parameters shared by every stage. */
#include "asr_model_params.h"

/*******************************************************************************/
/*                                PUBLIC DEFINES                               */
/*******************************************************************************/

/** @brief Input frames per output step. */
#define NET_FRAMES_PER_STEP (ASR_NETWORK_INPUT_FRAMES / ASR_NETWORK_OUTPUT_STEPS)

/** @brief Output steps of the left context. */
#define NET_LEFT_CONTEXT_STEPS (ASR_MODEL_NET_LEFT_CONTEXT_FRAMES / NET_FRAMES_PER_STEP)

/** @brief Output steps of the core, the part of a window decoded as final text. */
#define NET_CORE_CONTEXT_STEPS (ASR_MODEL_NET_STRIDE_FRAMES / NET_FRAMES_PER_STEP)

/** @brief Output steps of the right context. */
#define NET_RIGHT_CONTEXT_STEPS (ASR_MODEL_NET_RIGHT_CONTEXT_FRAMES / NET_FRAMES_PER_STEP)

/** @brief Blank output steps that separate two words. */
#define NET_PAUSE_SPACE_STEPS (ASR_MODEL_NET_PAUSE_SPACE_FRAMES / NET_FRAMES_PER_STEP)

/** @brief Blank output steps after a word that commit the mark seen at the previous window's end. */
#define NET_END_MARK_STEPS (ASR_MODEL_NET_END_MARK_FRAMES / NET_FRAMES_PER_STEP)

/** @brief Bytes of the int8 input tensor, @c [frame][band]. */
#define NET_INPUT_BYTES (ASR_NETWORK_INPUT_FRAMES * ASR_MODEL_MEL_BANDS)

/** @brief Bytes of the int8 output tensor, @c [step][class]. */
#define NET_OUTPUT_BYTES (ASR_NETWORK_OUTPUT_STEPS * ASR_NETWORK_OUTPUT_CLASSES)

/** @brief D-cache line of the Cortex-M55; cache maintenance works in whole lines. */
#define NET_CACHE_LINE_BYTES 32U

/** @brief Offset of the scratch area after the input tensor, rounded up to a cache line. */
#define NET_SCRATCH_OFFSET \
    (((NET_INPUT_BYTES + NET_CACHE_LINE_BYTES) - 1U) / NET_CACHE_LINE_BYTES * NET_CACHE_LINE_BYTES)

/** @brief Bytes of the input pool after the input tensor, free until the first inference. */
#define NET_SCRATCH_BYTES (ASR_NETWORK_INPUT_POOL_BYTES - NET_SCRATCH_OFFSET)

/**
 * @brief Characters one decode can produce, terminator included.
 * @note One per step, plus a space after blanks of the previous decode. With punctuation classes a
 *       mark and the letter after it can alternate, three characters for two steps (", A").
 */
#if defined(ASR_MODEL_NET_PUNCT_FIRST_INDEX)
#define NET_TEXT_CAPACITY (((3U * ASR_NETWORK_OUTPUT_STEPS) / 2U) + 2U)
#else
#define NET_TEXT_CAPACITY (ASR_NETWORK_OUTPUT_STEPS + 2U)
#endif

/**
 * @brief Decoder: 0 = greedy (default), 1 = CTC prefix beam search (net_ctc_beam.c).
 * @note  Measured on the host with this decoder code and the model's real int8 windows
 *        (cynipsnet_re/fwtest, 100 LibriSpeech dev-clean utterances): greedy 5.35 % WER, beam
 *        search 11.73 % (11.98 % without hotwords), mostly a space wrongly inserted after the
 *        first letter of an utterance, at 12x the CPU time. Keep 0 until the beam search is fixed
 *        and beats greedy on that bench.
 */
#ifndef ASR_CONF_DECODER_BEAM_SEARCH
#define ASR_CONF_DECODER_BEAM_SEARCH 0
#endif

#endif /* ASR_ENGINE_NET_NET_CFG_H */
