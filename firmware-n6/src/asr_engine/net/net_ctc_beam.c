/**
 * @file     net_ctc_beam.c
 * @brief    Embedded CTC prefix beam search decoder with hotword grammar for Cortex-M55.
 * @author   Antigravity & Nikita Prismakov
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "net_ctc_beam.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

#define NEG_INF (-1e30f)
#define OUTPUT_SCALE 0.267144024f

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

static net_hotword_entry_t hotword_table[NET_MAX_HOTWORDS];
static uint32_t hotword_count = 0U;

/** @brief Default embedded domain command hotwords. */
static const char *const default_hotwords[] = {
    "YES", "NO", "START", "STOP", "CANCEL", "CONFIRM", "PAUSE", "RESUME",
    "VOLUME UP", "VOLUME DOWN", "MUTE", "UNMUTE", "LIGHTS",
    "TURN ON", "TURN OFF", "TEMPERATURE", "HELP"
};

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

static inline float logsumexp_f(float a, float b)
{
    if (a <= NEG_INF / 2.0f)
    {
        return b;
    }
    if (b <= NEG_INF / 2.0f)
    {
        return a;
    }
    if (a > b)
    {
        return a + log1pf(expf(b - a));
    }
    return b + log1pf(expf(a - b));
}

/**
 * @brief Compute log-probabilities for 29 classes from int8 logits.
 */
static void compute_log_probs(const int8_t *row, float *log_probs)
{
    int8_t max_val = row[0];
    float sum_exp = 0.0f;
    uint32_t c;

    for (c = 1U; c < ASR_NETWORK_OUTPUT_CLASSES; ++c)
    {
        if (row[c] > max_val)
        {
            max_val = row[c];
        }
    }

    for (c = 0U; c < ASR_NETWORK_OUTPUT_CLASSES; ++c)
    {
        float diff = (float)(row[c] - max_val) * OUTPUT_SCALE;
        float exp_val = expf(diff);
        log_probs[c] = exp_val;
        sum_exp += exp_val;
    }

    float log_sum = logf(sum_exp > 1e-12f ? sum_exp : 1e-12f);
    for (c = 0U; c < ASR_NETWORK_OUTPUT_CLASSES; ++c)
    {
        float diff = (float)(row[c] - max_val) * OUTPUT_SCALE;
        log_probs[c] = diff - log_sum;
    }
}

/**
 * @brief Extract top-K candidate classes per frame, plus blank and space.
 */
static uint32_t get_top_candidates(const float *log_probs, uint32_t *candidates, uint32_t max_cand)
{
    uint32_t cand_count = 0U;
    bool selected[ASR_NETWORK_OUTPUT_CLASSES] = {false};

    /* Always include BLANK and SPACE */
    candidates[cand_count++] = ASR_MODEL_NET_BLANK_INDEX;
    selected[ASR_MODEL_NET_BLANK_INDEX] = true;

    candidates[cand_count++] = ASR_MODEL_NET_SPACE_INDEX;
    selected[ASR_MODEL_NET_SPACE_INDEX] = true;

    while (cand_count < max_cand)
    {
        float best_prob = NEG_INF;
        int32_t best_c = -1;
        uint32_t c;

        for (c = 0U; c < ASR_NETWORK_OUTPUT_CLASSES; ++c)
        {
            if (!selected[c] && (log_probs[c] > best_prob))
            {
                best_prob = log_probs[c];
                best_c = (int32_t)c;
            }
        }

        if (best_c < 0)
        {
            break;
        }

        selected[best_c] = true;
        candidates[cand_count++] = (uint32_t)best_c;
    }

    return cand_count;
}

/**
 * @brief Check if the prefix tail matches or extends any registered hotword or phrase.
 */
static float check_hotword_bonus(const char *text, uint16_t len, bool is_word_boundary, float hw_weight, float pfx_weight)
{
    if ((text == NULL) || (len == 0U))
    {
        return 0.0f;
    }

    /* Find trailing text excluding trailing space */
    int32_t end = (int32_t)len;
    while ((end > 0) && (text[end - 1] == ' '))
    {
        --end;
    }
    if (end <= 0)
    {
        return 0.0f;
    }

    uint32_t h;
    for (h = 0U; h < hotword_count; ++h)
    {
        const char *hw = hotword_table[h].word;
        size_t hw_len = strlen(hw);
        float weight = (hotword_table[h].weight > 0.0f) ? hotword_table[h].weight : hw_weight;

        if (is_word_boundary)
        {
            /* Full phrase match: check if text ends with hw preceded by start-of-string or space */
            if ((size_t)end >= hw_len)
            {
                size_t start = (size_t)end - hw_len;
                if ((start == 0U) || (text[start - 1U] == ' '))
                {
                    if (strncmp(&text[start], hw, hw_len) == 0)
                    {
                        return weight;
                    }
                }
            }
        }
        else
        {
            /* Partial prefix match: check if the current trailing phrase/word is a prefix of hw */
            int32_t s;
            for (s = end - 1; s >= 0; --s)
            {
                if ((s == 0) || (text[s - 1] == ' '))
                {
                    size_t seg_len = (size_t)(end - s);
                    if ((seg_len > 0U) && (seg_len <= hw_len))
                    {
                        if (strncmp(&text[s], hw, seg_len) == 0)
                        {
                            return pfx_weight;
                        }
                    }
                }
            }
        }
    }
    return 0.0f;
}

/**
 * @brief Find existing candidate in pool matching text, or allocate a new slot.
 */
static uint32_t find_or_add_candidate(net_beam_entry_t *pool, uint32_t *pool_count,
                                      const char *text, uint16_t length, uint16_t start_len,
                                      uint16_t pause_steps, float lm_score)
{
    uint32_t j;
    uint32_t count = *pool_count;

    for (j = 0U; j < count; ++j)
    {
        if ((pool[j].length == length) && (strcmp(pool[j].text, text) == 0))
        {
            return j;
        }
    }

    if (count < NET_BEAM_MAX_CANDIDATES)
    {
        memcpy(pool[count].text, text, (size_t)length + 1U);
        pool[count].length = length;
        pool[count].start_len = start_len;
        pool[count].pause_steps = pause_steps;
        pool[count].log_p_b = NEG_INF;
        pool[count].log_p_nb = NEG_INF;
        pool[count].lm_score = lm_score;
        pool[count].total_score = NEG_INF;
        *pool_count = count + 1U;
        return count;
    }

    return 0xFFFFFFFFU;
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void net_ctc_beam_init(net_beam_state_t *state, uint32_t beam_width, float hotword_weight)
{
    if (state == NULL)
    {
        return;
    }

    if (beam_width == 0U)
    {
        beam_width = 1U;
    }
    if (beam_width > NET_BEAM_MAX_WIDTH)
    {
        beam_width = NET_BEAM_MAX_WIDTH;
    }

    state->beam_width = beam_width;
    state->hotword_weight = hotword_weight > 0.0f ? hotword_weight : NET_DEFAULT_HOTWORD_WEIGHT;
    state->prefix_weight = NET_DEFAULT_PREFIX_WEIGHT;

    net_ctc_beam_reset(state);

    /* Initialize default hotwords if none registered yet */
    if (hotword_count == 0U)
    {
        uint32_t i;
        uint32_t n = sizeof(default_hotwords) / sizeof(default_hotwords[0]);
        for (i = 0U; (i < n) && (i < NET_MAX_HOTWORDS); ++i)
        {
            (void)net_ctc_beam_add_hotword(default_hotwords[i], state->hotword_weight);
        }
    }
}

void net_ctc_beam_reset(net_beam_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    state->count = 1U;
    state->entries[0].text[0] = '\0';
    state->entries[0].length = 0U;
    state->entries[0].start_len = 0U;
    state->entries[0].pause_steps = 0U;
    state->entries[0].log_p_b = 0.0f;
    state->entries[0].log_p_nb = NEG_INF;
    state->entries[0].lm_score = 0.0f;
    state->entries[0].total_score = 0.0f;
}

bool net_ctc_beam_add_hotword(const char *word, float weight)
{
    if ((word == NULL) || (hotword_count >= NET_MAX_HOTWORDS))
    {
        return false;
    }

    size_t len = strlen(word);
    if ((len == 0U) || (len >= NET_MAX_HOTWORD_LEN))
    {
        return false;
    }

    strncpy(hotword_table[hotword_count].word, word, NET_MAX_HOTWORD_LEN - 1U);
    hotword_table[hotword_count].word[NET_MAX_HOTWORD_LEN - 1U] = '\0';
    hotword_table[hotword_count].weight = weight;
    ++hotword_count;
    return true;
}

void net_ctc_beam_clear_hotwords(void)
{
    hotword_count = 0U;
}

uint32_t net_ctc_beam_decode(net_beam_state_t *state,
                            const int8_t *logits,
                            uint32_t first_step,
                            uint32_t steps,
                            char *text,
                            uint32_t capacity)
{
    if ((text == NULL) || (capacity == 0U))
    {
        return 0U;
    }
    text[0] = '\0';

    if ((state == NULL) || (logits == NULL) ||
        (first_step > ASR_NETWORK_OUTPUT_STEPS) || (steps > (ASR_NETWORK_OUTPUT_STEPS - first_step)))
    {
        return 0U;
    }

    if (state->count == 0U)
    {
        net_ctc_beam_reset(state);
    }

    /* Record start length for each hypothesis at the beginning of this decode window */
    uint32_t b;
    for (b = 0U; b < state->count; ++b)
    {
        state->entries[b].start_len = state->entries[b].length;
    }

    /* Candidate buffer for next step expansion */
    static net_beam_entry_t next_pool[NET_BEAM_MAX_CANDIDATES];
    uint32_t next_count = 0U;

    float log_probs[ASR_NETWORK_OUTPUT_CLASSES];
    uint32_t candidates[NET_BEAM_TOP_K + 2U];
    uint32_t step;

    for (step = first_step; step < (first_step + steps); ++step)
    {
        const int8_t *row = &logits[step * ASR_NETWORK_OUTPUT_CLASSES];
        compute_log_probs(row, log_probs);

        uint32_t cand_count = get_top_candidates(log_probs, candidates, NET_BEAM_TOP_K + 2U);
        next_count = 0U;

        for (b = 0U; b < state->count; ++b)
        {
            const net_beam_entry_t *entry = &state->entries[b];
            float p_tot = logsumexp_f(entry->log_p_b, entry->log_p_nb);
            char last_char = (entry->length > 0U) ? entry->text[entry->length - 1U] : '\0';

            uint32_t k;
            for (k = 0U; k < cand_count; ++k)
            {
                uint32_t c = candidates[k];
                float lp = log_probs[c];

                if (c == ASR_MODEL_NET_BLANK_INDEX)
                {
                    /* Blank preserves prefix */
                    uint16_t new_pause = (entry->pause_steps < NET_PAUSE_SPACE_STEPS) ?
                                         (uint16_t)(entry->pause_steps + 1U) : (uint16_t)NET_PAUSE_SPACE_STEPS;
                    uint32_t idx = find_or_add_candidate(next_pool, &next_count,
                                                         entry->text, entry->length, entry->start_len,
                                                         new_pause, entry->lm_score);
                    if (idx != 0xFFFFFFFFU)
                    {
                        next_pool[idx].log_p_b = logsumexp_f(next_pool[idx].log_p_b, p_tot + lp);
                        if (new_pause > next_pool[idx].pause_steps)
                        {
                            next_pool[idx].pause_steps = new_pause;
                        }
                    }
                }
                else if (c == ASR_MODEL_NET_SPACE_INDEX)
                {
                    /* Space token */
                    if ((last_char == ' ') || (last_char == '\0'))
                    {
                        /* Do not duplicate spaces or start with space */
                        uint32_t idx = find_or_add_candidate(next_pool, &next_count,
                                                             entry->text, entry->length, entry->start_len,
                                                             0U, entry->lm_score);
                        if (idx != 0xFFFFFFFFU)
                        {
                            next_pool[idx].log_p_nb = logsumexp_f(next_pool[idx].log_p_nb, p_tot + lp);
                            next_pool[idx].pause_steps = 0U;
                        }
                    }
                    else if ((entry->length + 1U) < NET_TEXT_CAPACITY)
                    {
                        char new_text[NET_TEXT_CAPACITY];
                        memcpy(new_text, entry->text, entry->length);
                        new_text[entry->length] = ' ';
                        new_text[entry->length + 1U] = '\0';
                        uint16_t new_len = entry->length + 1U;

                        float bonus = check_hotword_bonus(new_text, new_len, true,
                                                          state->hotword_weight, state->prefix_weight);
                        uint32_t idx = find_or_add_candidate(next_pool, &next_count,
                                                             new_text, new_len, entry->start_len,
                                                             0U, entry->lm_score + bonus);
                        if (idx != 0xFFFFFFFFU)
                        {
                            next_pool[idx].log_p_nb = logsumexp_f(next_pool[idx].log_p_nb, p_tot + lp);
                        }
                    }
                }
                else
                {
                    /* Letter character token */
                    char ch = ASR_MODEL_NET_TOKENS[c];
                    bool needs_pause_space = (entry->pause_steps >= NET_PAUSE_SPACE_STEPS) &&
                                             (last_char != ' ') && (last_char != '\0');

                    if (needs_pause_space)
                    {
                        /* Insert pause-space before new word */
                        if ((entry->length + 2U) < NET_TEXT_CAPACITY)
                        {
                            char new_text[NET_TEXT_CAPACITY];
                            memcpy(new_text, entry->text, entry->length);
                            new_text[entry->length] = ' ';
                            new_text[entry->length + 1U] = ch;
                            new_text[entry->length + 2U] = '\0';
                            uint16_t new_len = entry->length + 2U;

                            float bonus = check_hotword_bonus(new_text, new_len, false,
                                                              state->hotword_weight, state->prefix_weight);
                            uint32_t idx = find_or_add_candidate(next_pool, &next_count,
                                                                 new_text, new_len, entry->start_len,
                                                                 0U, entry->lm_score + bonus);
                            if (idx != 0xFFFFFFFFU)
                            {
                                next_pool[idx].log_p_nb = logsumexp_f(next_pool[idx].log_p_nb, p_tot + lp);
                            }
                        }
                    }
                    else if (ch == last_char)
                    {
                        /* 1. Repeat without blank collapses to same prefix */
                        uint32_t idx_c = find_or_add_candidate(next_pool, &next_count,
                                                               entry->text, entry->length, entry->start_len,
                                                               0U, entry->lm_score);
                        if (idx_c != 0xFFFFFFFFU)
                        {
                            next_pool[idx_c].log_p_nb = logsumexp_f(next_pool[idx_c].log_p_nb, entry->log_p_nb + lp);
                        }

                        /* 2. Repeat with blank extends prefix */
                        if ((entry->length + 1U) < NET_TEXT_CAPACITY)
                        {
                            char new_text[NET_TEXT_CAPACITY];
                            memcpy(new_text, entry->text, entry->length);
                            new_text[entry->length] = ch;
                            new_text[entry->length + 1U] = '\0';
                            uint16_t new_len = entry->length + 1U;

                            float bonus = check_hotword_bonus(new_text, new_len, false,
                                                              state->hotword_weight, state->prefix_weight);
                            uint32_t idx_e = find_or_add_candidate(next_pool, &next_count,
                                                                   new_text, new_len, entry->start_len,
                                                                   0U, entry->lm_score + bonus);
                            if (idx_e != 0xFFFFFFFFU)
                            {
                                next_pool[idx_e].log_p_nb = logsumexp_f(next_pool[idx_e].log_p_nb, entry->log_p_b + lp);
                            }
                        }
                    }
                    else
                    {
                        /* Distinct character extends prefix */
                        if ((entry->length + 1U) < NET_TEXT_CAPACITY)
                        {
                            char new_text[NET_TEXT_CAPACITY];
                            memcpy(new_text, entry->text, entry->length);
                            new_text[entry->length] = ch;
                            new_text[entry->length + 1U] = '\0';
                            uint16_t new_len = entry->length + 1U;

                            float bonus = check_hotword_bonus(new_text, new_len, false,
                                                              state->hotword_weight, state->prefix_weight);
                            uint32_t idx = find_or_add_candidate(next_pool, &next_count,
                                                                 new_text, new_len, entry->start_len,
                                                                 0U, entry->lm_score + bonus);
                            if (idx != 0xFFFFFFFFU)
                            {
                                next_pool[idx].log_p_nb = logsumexp_f(next_pool[idx].log_p_nb, p_tot + lp);
                            }
                        }
                    }
                }
            }
        }

        /* Calculate total score for all candidates */
        uint32_t i;
        for (i = 0U; i < next_count; ++i)
        {
            next_pool[i].total_score = logsumexp_f(next_pool[i].log_p_b, next_pool[i].log_p_nb) + next_pool[i].lm_score;
        }

        /* Prune to beam_width */
        state->count = (next_count < state->beam_width) ? next_count : state->beam_width;

        /* Selection sort top entries into state */
        for (b = 0U; b < state->count; ++b)
        {
            uint32_t best_idx = b;
            for (i = b + 1U; i < next_count; ++i)
            {
                if (next_pool[i].total_score > next_pool[best_idx].total_score)
                {
                    best_idx = i;
                }
            }
            if (best_idx != b)
            {
                net_beam_entry_t tmp = next_pool[b];
                next_pool[b] = next_pool[best_idx];
                next_pool[best_idx] = tmp;
            }
            state->entries[b] = next_pool[b];
        }
    }

    /* Output only the delta characters produced in THIS decode call */
    uint32_t delta_written = 0U;
    if (state->count > 0U)
    {
        const net_beam_entry_t *winner = &state->entries[0];
        if (winner->length > winner->start_len)
        {
            uint32_t delta_len = (uint32_t)(winner->length - winner->start_len);
            delta_written = (delta_len < capacity) ? delta_len : (capacity - 1U);
            memcpy(text, &winner->text[winner->start_len], delta_written);
            text[delta_written] = '\0';
        }
        else
        {
            text[0] = '\0';
            delta_written = 0U;
        }

        /* History Management: trim already-committed prefix words older than current word */
        if (winner->length > 32U)
        {
            /* Look for last space before the active tail */
            int32_t trim_pos = -1;
            int32_t p;
            for (p = (int32_t)winner->length - 1; p >= 0; --p)
            {
                if ((winner->text[p] == ' ') && (((int32_t)winner->length - 1 - p) >= 12))
                {
                    trim_pos = p;
                    break;
                }
            }
            if (trim_pos > 0)
            {
                uint32_t trim_bytes = (uint32_t)(trim_pos + 1);
                for (b = 0U; b < state->count; ++b)
                {
                    net_beam_entry_t *e = &state->entries[b];
                    if (e->length >= trim_bytes)
                    {
                        uint16_t rem = (uint16_t)(e->length - (uint16_t)trim_bytes);
                        memmove(e->text, &e->text[trim_bytes], (size_t)rem + 1U);
                        e->length = rem;
                    }
                    else
                    {
                        e->text[0] = '\0';
                        e->length = 0U;
                    }
                    e->start_len = e->length;
                }
            }
        }
        else
        {
            /* Update start_len for next decode call */
            for (b = 0U; b < state->count; ++b)
            {
                state->entries[b].start_len = state->entries[b].length;
            }
        }

        return delta_written;
    }

    return 0U;
}
