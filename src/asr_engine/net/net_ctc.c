/**
 * @file     net_ctc.c
 * @brief    Streaming greedy CTC decoding; implements net_ctc.h.
 * @author   Nikita Prismakov
 *
 * SPDX-FileCopyrightText: 2026 Nikita Prismakov
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "net_ctc.h"

#include <stddef.h>

/*******************************************************************************/
/*                              STATIC ASSERTIONS                              */
/*******************************************************************************/

_Static_assert((sizeof(ASR_MODEL_NET_TOKENS) - 1U) == ASR_NETWORK_OUTPUT_CLASSES,
               "ASR_MODEL_NET_TOKENS needs exactly one character per output class");
_Static_assert(ASR_MODEL_NET_BLANK_INDEX < ASR_NETWORK_OUTPUT_CLASSES,
               "the blank must be one of the classes");
_Static_assert((ASR_MODEL_NET_SPACE_INDEX < ASR_NETWORK_OUTPUT_CLASSES) &&
                   (ASR_MODEL_NET_SPACE_INDEX != ASR_MODEL_NET_BLANK_INDEX),
               "the space must be one of the printed classes");
_Static_assert(((ASR_MODEL_NET_PAUSE_SPACE_FRAMES % NET_FRAMES_PER_STEP) == 0U) &&
               (NET_PAUSE_SPACE_STEPS > 0U),
               "ASR_MODEL_NET_PAUSE_SPACE_FRAMES must be a whole number of output steps, "
               "at least one (NET_TEXT_CAPACITY counts on it)");
#if defined(ASR_MODEL_NET_PUNCT_FIRST_INDEX)
_Static_assert(((ASR_MODEL_NET_END_MARK_FRAMES % NET_FRAMES_PER_STEP) == 0U) && (NET_END_MARK_STEPS > 0U),
               "ASR_MODEL_NET_END_MARK_FRAMES must be a whole number of output steps, at least one");
_Static_assert((ASR_MODEL_NET_PUNCT_FIRST_INDEX > ASR_MODEL_NET_SPACE_INDEX) &&
               (ASR_MODEL_NET_PUNCT_FIRST_INDEX < ASR_NETWORK_OUTPUT_CLASSES),
               "the marks are the last classes, after the letters");
#endif
_Static_assert((ASR_NETWORK_INPUT_FRAMES % ASR_NETWORK_OUTPUT_STEPS) == 0U,
               "the network must reduce the frame rate by a whole factor");
_Static_assert(ASR_MODEL_NET_STRIDE_FRAMES > 0U, "the core must advance the window");
_Static_assert((ASR_MODEL_NET_LEFT_CONTEXT_FRAMES + ASR_MODEL_NET_STRIDE_FRAMES +
                ASR_MODEL_NET_RIGHT_CONTEXT_FRAMES) == ASR_NETWORK_INPUT_FRAMES,
               "ASR_MODEL_NET_LEFT_CONTEXT_FRAMES + ASR_MODEL_NET_STRIDE_FRAMES + "
               "ASR_MODEL_NET_RIGHT_CONTEXT_FRAMES must equal the receptive field "
               "(ASR_NETWORK_INPUT_FRAMES)");
_Static_assert(((ASR_MODEL_NET_LEFT_CONTEXT_FRAMES % NET_FRAMES_PER_STEP) == 0U) &&
               ((ASR_MODEL_NET_STRIDE_FRAMES % NET_FRAMES_PER_STEP) == 0U) &&
               ((ASR_MODEL_NET_RIGHT_CONTEXT_FRAMES % NET_FRAMES_PER_STEP) == 0U),
               "left context, core and right context must be whole output steps");

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

#if !defined(ASR_CONF_DECODER_BEAM_SEARCH) || (ASR_CONF_DECODER_BEAM_SEARCH == 0)
/**
 * @brief Return the most likely class of one output step.
 * @param[in] row @c ASR_NETWORK_OUTPUT_CLASSES int8 logits.
 * @return Index of the maximum; the lowest index wins a tie.
 */
static int32_t most_likely_class(const int8_t *row)
{
    int32_t best = 0;
    uint32_t index;

    for (index = 1U; index < ASR_NETWORK_OUTPUT_CLASSES; ++index)
    {
        if (row[index] > row[best])
        {
            best = (int32_t)index;
        }
    }
    return best;
}
#endif

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void net_ctc_add_hotword(const char *word, float weight)
{
    (void)net_ctc_beam_add_hotword(word, weight);
}

void net_ctc_clear_hotwords(void)
{
    net_ctc_beam_clear_hotwords();
}

void net_ctc_reset(net_ctc_state_t *state)
{
    if (state != NULL)
    {
        state->previous_class = -1;
        state->printed_class = -1;
        state->pause_steps = 0U;
        state->pending_space = 0U;
        state->edge_mark = -1;
        state->silence_steps = 0U;
#if defined(ASR_CONF_DECODER_BEAM_SEARCH) && (ASR_CONF_DECODER_BEAM_SEARCH != 0)
        net_ctc_beam_init(&state->beam, NET_BEAM_MAX_WIDTH, NET_DEFAULT_HOTWORD_WEIGHT);
#endif
    }
}

uint32_t net_ctc_decode(net_ctc_state_t *state,
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

#if defined(ASR_CONF_DECODER_BEAM_SEARCH) && (ASR_CONF_DECODER_BEAM_SEARCH != 0)
    return net_ctc_beam_decode(&state->beam, logits, first_step, steps, text, capacity);
#else
    uint32_t length = 0U;
    uint32_t step;

    for (step = first_step; step < (first_step + steps); ++step)
    {
        int32_t best = most_likely_class(&logits[step * ASR_NETWORK_OUTPUT_CLASSES]);
        int paused;
        int needs_space;

        if (best == (int32_t)ASR_MODEL_NET_BLANK_INDEX)
        {
            /* Silence: count it only as far as it takes to make a pause. */
            if (state->pause_steps < NET_PAUSE_SPACE_STEPS)
            {
                ++state->pause_steps;
            }
#if defined(ASR_MODEL_NET_PUNCT_FIRST_INDEX)
            /* The speaker stopped after a word the network had ended at the previous window's
               edge: commit that mark now (see ASR_MODEL_NET_END_MARK_FRAMES). */
            if (state->silence_steps < NET_END_MARK_STEPS)
            {
                ++state->silence_steps;
                if ((state->silence_steps == NET_END_MARK_STEPS) && (state->edge_mark >= 0) &&
                    (state->printed_class > (int32_t)ASR_MODEL_NET_SPACE_INDEX) &&
                    (state->printed_class < (int32_t)ASR_MODEL_NET_PUNCT_FIRST_INDEX))
                {
                    state->printed_class = state->edge_mark;
                    state->pending_space = 1U;
                    if ((length + 1U) < capacity)
                    {
                        text[length] = ASR_MODEL_NET_TOKENS[state->edge_mark];
                        ++length;
                    }
                    state->edge_mark = -1;
                }
            }
#endif
            state->previous_class = best;
            continue;
        }
        paused = (state->pause_steps >= NET_PAUSE_SPACE_STEPS);
        state->pause_steps = 0U;
        state->silence_steps = 0U;

        /* A repeat still belongs to the token already seen. */
        if (best == state->previous_class)
        {
            continue;
        }
        state->previous_class = best;

#if defined(ASR_MODEL_NET_PUNCT_FIRST_INDEX)
        /* With marks, a space is held back until the next letter: a mark that comes
           first then sticks to its word ("WORD," not "WORD ,"). */
        if (best == (int32_t)ASR_MODEL_NET_SPACE_INDEX)
        {
            if (state->printed_class >= 0)
            {
                state->pending_space = 1U;
            }
            continue;
        }
        if (best >= (int32_t)ASR_MODEL_NET_PUNCT_FIRST_INDEX)
        {
            /* Only a word can be ended. */
            if (state->printed_class < 0)
            {
                continue;
            }
            /* Ended once already: a stronger mark (? over . over , by class order) replaces
               the first while it is still in this text; at a sentence end the network often
               gives a comma and then the full stop. */
            if (state->printed_class >= (int32_t)ASR_MODEL_NET_PUNCT_FIRST_INDEX)
            {
                if ((best > state->printed_class) && (length > 0U) &&
                    (text[length - 1U] == ASR_MODEL_NET_TOKENS[state->printed_class]))
                {
                    text[length - 1U] = ASR_MODEL_NET_TOKENS[best];
                    state->printed_class = best;
                }
                continue;
            }
            state->printed_class = best;
            state->pending_space = 1U;
            if ((length + 1U) < capacity)
            {
                text[length] = ASR_MODEL_NET_TOKENS[best];
                ++length;
            }
            continue;
        }
        needs_space = (state->pending_space || paused) && (state->printed_class >= 0);
        state->pending_space = 0U;
#else
        /* A pause between two characters separates two words, unless a space
           already does. */
        needs_space = paused && (state->printed_class >= 0) &&
                      (state->printed_class != (int32_t)ASR_MODEL_NET_SPACE_INDEX) &&
                      (best != (int32_t)ASR_MODEL_NET_SPACE_INDEX);
#endif
        state->printed_class = best;

        /* A character that does not fit with the terminator is dropped. */
        if (needs_space && ((length + 1U) < capacity))
        {
            text[length] = ASR_MODEL_NET_TOKENS[ASR_MODEL_NET_SPACE_INDEX];
            ++length;
        }
        if ((length + 1U) < capacity)
        {
            text[length] = ASR_MODEL_NET_TOKENS[best];
            ++length;
        }
    }
    text[length] = '\0';
    return length;
#endif
}

void net_ctc_decode_window(net_ctc_state_t *state,
                           const int8_t *logits,
                           int first_window,
                           char *committed,
                           uint32_t committed_capacity,
                           char *tentative,
                           uint32_t tentative_capacity)
{
    const uint32_t committed_end = NET_LEFT_CONTEXT_STEPS + NET_CORE_CONTEXT_STEPS;
    const uint32_t committed_start = first_window ? 0U : NET_LEFT_CONTEXT_STEPS;
    /* Static, not on the stack: with the beam search the state is over 5 KB, more than the whole
       4 KB stack of the network thread. Only that thread decodes, so one copy is enough. */
    static net_ctc_state_t lookahead;

    (void)net_ctc_decode(state,
                         logits,
                         committed_start,
                         committed_end - committed_start,
                         committed,
                         committed_capacity);

    /* Decode the right context on a copy: the next window decodes it again as its core. */
    if (state == NULL)
    {
        (void)net_ctc_decode(NULL, logits, 0U, 0U, tentative, tentative_capacity);
        return;
    }
    lookahead = *state;
#if defined(ASR_MODEL_NET_PUNCT_FIRST_INDEX)
    {
        uint32_t length;

        lookahead.edge_mark = -1;
        length = net_ctc_decode(&lookahead,
                                logits,
                                committed_end,
                                NET_RIGHT_CONTEXT_STEPS,
                                tentative,
                                tentative_capacity);

        /* The network tends to end a sentence where the window ends, its view of the end of
           the audio. Such a mark is not shown; it is kept, and committed if the speaker has
           indeed stopped (net_ctc_decode). */
        state->edge_mark = -1;
        if ((length > 0U) && (length < tentative_capacity))
        {
            const char last = tentative[length - 1U];
            uint32_t mark;

            for (mark = ASR_MODEL_NET_PUNCT_FIRST_INDEX; mark < ASR_NETWORK_OUTPUT_CLASSES; ++mark)
            {
                if (last == ASR_MODEL_NET_TOKENS[mark])
                {
                    tentative[length - 1U] = '\0';
                    state->edge_mark = (int32_t)mark;
                    break;
                }
            }
        }
    }
#else
    (void)net_ctc_decode(&lookahead,
                         logits,
                         committed_end,
                         NET_RIGHT_CONTEXT_STEPS,
                         tentative,
                         tentative_capacity);
#endif
}
