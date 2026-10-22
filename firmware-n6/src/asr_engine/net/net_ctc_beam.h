/**
 * @file     net_ctc_beam.h
 * @brief    CTC prefix beam search decoder with hotword grammar for Cortex-M55.
 * @author   Antigravity & Nikita Prismakov
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ASR_ENGINE_NET_NET_CTC_BEAM_H
#define ASR_ENGINE_NET_NET_CTC_BEAM_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>
#include <stdbool.h>
#include "net_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                                PUBLIC DEFINES                               */
/*******************************************************************************/

/** @brief Maximum beam width (active prefix hypotheses). */
#define NET_BEAM_MAX_WIDTH 16U

/** @brief Top candidate tokens per step (+ blank & space). */
#define NET_BEAM_TOP_K 8U

/** @brief Maximum candidate hypotheses evaluated per step. */
#define NET_BEAM_MAX_CANDIDATES (NET_BEAM_MAX_WIDTH * (NET_BEAM_TOP_K + 2U))

/** @brief Maximum number of registered domain hotwords. */
#define NET_MAX_HOTWORDS 24U

/** @brief Maximum length of a hotword string. */
#define NET_MAX_HOTWORD_LEN 24U

/** @brief Default hotword log-probability bonus. */
#define NET_DEFAULT_HOTWORD_WEIGHT 3.5f

/** @brief Default partial prefix incremental bonus. */
#define NET_DEFAULT_PREFIX_WEIGHT 0.5f

/*******************************************************************************/
/*                                 PUBLIC TYPES                                */
/*******************************************************************************/

/** @brief Single prefix hypothesis in the beam. */
typedef struct
{
    /** @brief NUL-terminated text string of the prefix. */
    char text[NET_TEXT_CAPACITY];
    /** @brief Number of characters in text. */
    uint16_t length;
    /** @brief Number of characters in text when current decode call started. */
    uint16_t start_len;
    /** @brief Blank steps accumulated since last non-blank token. */
    uint16_t pause_steps;
    /** @brief Log-probability that prefix ends in blank. */
    float log_p_b;
    /** @brief Log-probability that prefix ends in non-blank. */
    float log_p_nb;
    /** @brief Cumulative language model / hotword score bonus. */
    float lm_score;
    /** @brief Total combined score: logsumexp(p_b, p_nb) + lm_score. */
    float total_score;
} net_beam_entry_t;

/** @brief Decoder state across frames and windows. */
typedef struct
{
    /** @brief Active hypotheses in the beam. */
    net_beam_entry_t entries[NET_BEAM_MAX_WIDTH];
    /** @brief Current number of active hypotheses. */
    uint32_t count;
    /** @brief Configured beam width. */
    uint32_t beam_width;
    /** @brief Hotword completion bonus weight. */
    float hotword_weight;
    /** @brief Partial prefix incremental bonus weight. */
    float prefix_weight;
} net_beam_state_t;

/** @brief Hotword descriptor. */
typedef struct
{
    char word[NET_MAX_HOTWORD_LEN];
    float weight;
} net_hotword_entry_t;

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief Initialize a beam search decoder state.
 * @param[out] state State to initialize.
 * @param[in] beam_width Active beam width (1 to NET_BEAM_MAX_WIDTH).
 * @param[in] hotword_weight Bonus for hotword completion.
 */
void net_ctc_beam_init(net_beam_state_t *state, uint32_t beam_width, float hotword_weight);

/**
 * @brief Reset beam state to empty prefix for a new stream.
 * @param[in,out] state State to reset.
 */
void net_ctc_beam_reset(net_beam_state_t *state);

/**
 * @brief Register a domain hotword to boost its detection probability.
 * @param[in] word Uppercase word string.
 * @param[in] weight Bonus weight (> 0).
 * @return true if added, false if table full.
 */
bool net_ctc_beam_add_hotword(const char *word, float weight);

/**
 * @brief Clear registered hotwords table.
 */
void net_ctc_beam_clear_hotwords(void);

/**
 * @brief Run CTC prefix beam search over a sequence of output steps.
 * @param[in,out] state Beam decoder state.
 * @param[in] logits Output logits array ([step][class] int8).
 * @param[in] first_step Index of the first step.
 * @param[in] steps Number of steps to decode.
 * @param[out] text Output buffer receiving the best hypothesis string.
 * @param[in] capacity Size of text buffer in bytes.
 * @return Number of characters written (terminator excluded).
 */
uint32_t net_ctc_beam_decode(net_beam_state_t *state,
                            const int8_t *logits,
                            uint32_t first_step,
                            uint32_t steps,
                            char *text,
                            uint32_t capacity);

#ifdef __cplusplus
}
#endif

#endif /* ASR_ENGINE_NET_NET_CTC_BEAM_H */
