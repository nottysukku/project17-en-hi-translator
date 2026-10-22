/**
 * @file     xl_voice_net.c
 * @brief    The flow and vocoder networks on the NPU; implements xl_voice_net.h.
 * @details  Built against the code ST Edge AI generates with --name flow and --name vocoder
 *           (stai_flow.h, stai_vocoder.h; CMake defines XL_HAVE_VOICE_NETWORKS when both are in
 *           src/tts/networks). The static assertions check that the generated networks are the
 *           graphs model-tools exports, with float32 inputs and outputs (st_compile.sh).
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xl_voice_net.h"

#include <stddef.h>

#if defined(XL_HAVE_VOICE_NETWORKS) && (XL_HAVE_VOICE_NETWORKS != 0)

#include "ll_aton_caches_interface.h"
#include "stai_flow.h"
#include "stai_vocoder.h"

/* flow: inputs latents [192 x 512] and mask [512], output [192 x 512], all float32 */
_Static_assert((STAI_FLOW_IN_NUM == 2) && (STAI_FLOW_OUT_NUM == 1), "flow: expected 2 inputs and 1 output");
_Static_assert((STAI_FLOW_IN_1_FORMAT == STAI_FORMAT_FLOAT32) && (STAI_FLOW_IN_2_FORMAT == STAI_FORMAT_FLOAT32) &&
                   (STAI_FLOW_OUT_1_FORMAT == STAI_FORMAT_FLOAT32),
               "flow: compile with --input-data-type float32 --output-data-type float32");
_Static_assert(STAI_FLOW_IN_1_SIZE == (XL_VOICE_CHANNELS * XL_VOICE_FLOW_FRAMES), "flow: input 1 is not 192 x 512");
_Static_assert(STAI_FLOW_IN_2_SIZE == XL_VOICE_FLOW_FRAMES, "flow: input 2 (mask) is not 512");
_Static_assert(STAI_FLOW_OUT_1_SIZE == (XL_VOICE_CHANNELS * XL_VOICE_FLOW_FRAMES), "flow: output is not 192 x 512");
_Static_assert(STAI_FLOW_IN_1_CHANNEL == XL_VOICE_CHANNELS, "flow: input channels are not the latent channels");
_Static_assert(STAI_FLOW_OUT_1_CHANNEL == XL_VOICE_CHANNELS, "flow: output channels are not the latent channels");

/* vocoder: input latents [192 x 94], output 16384 samples, float32 */
_Static_assert((STAI_VOCODER_IN_NUM == 1) && (STAI_VOCODER_OUT_NUM == 1), "vocoder: expected 1 input and 1 output");
_Static_assert((STAI_VOCODER_IN_1_FORMAT == STAI_FORMAT_FLOAT32) && (STAI_VOCODER_OUT_1_FORMAT == STAI_FORMAT_FLOAT32),
               "vocoder: compile with --input-data-type float32 --output-data-type float32");
_Static_assert(STAI_VOCODER_IN_1_SIZE == (XL_VOICE_CHANNELS * XL_VOICE_VOC_IN_FRAMES),
               "vocoder: input is not 192 x 94 (tts_export.py ... 64 t)");
_Static_assert(STAI_VOCODER_IN_1_CHANNEL == XL_VOICE_CHANNELS, "vocoder: input channels are not the latent channels");
_Static_assert(STAI_VOCODER_OUT_1_SIZE == (XL_VOICE_VOC_FRAMES * 256U), "vocoder: output is not 64 x 256 samples");

/** @brief Strides of a [channels x frames] tensor in the layout the generated code uses. */
#define XL_STRIDES(flags, frames)                                                                                      \
    ((((flags) & STAI_FLAG_CHANNEL_LAST) != 0U) ? 1U : (frames)),                                                      \
        ((((flags) & STAI_FLAG_CHANNEL_LAST) != 0U) ? XL_VOICE_CHANNELS : 1U)

static STAI_NETWORK_CONTEXT_DECLARE(flow_context, STAI_FLOW_CONTEXT_SIZE)
static STAI_NETWORK_CONTEXT_DECLARE(vocoder_context, STAI_VOCODER_CONTEXT_SIZE)

static float *flow_in;
static float *flow_mask;
static float *flow_out;
static float *voc_in;
static float *voc_out;
static int ready;

int xl_voice_net_init(void)
{
    stai_ptr in[2], out[1];
    stai_size n = 0;

    if (ready)
    {
        return 0;
    }
    if ((stai_flow_init(flow_context) != STAI_SUCCESS) || (stai_vocoder_init(vocoder_context) != STAI_SUCCESS))
    {
        return -1;
    }
    if ((stai_flow_get_inputs(flow_context, in, &n) != STAI_SUCCESS) || (n != 2U))
    {
        return -2;
    }
    flow_in = (float *)in[0];
    flow_mask = (float *)in[1];
    if ((stai_flow_get_outputs(flow_context, out, &n) != STAI_SUCCESS) || (n != 1U))
    {
        return -3;
    }
    flow_out = (float *)out[0];
    if ((stai_vocoder_get_inputs(vocoder_context, in, &n) != STAI_SUCCESS) || (n != 1U))
    {
        return -4;
    }
    voc_in = (float *)in[0];
    if ((stai_vocoder_get_outputs(vocoder_context, out, &n) != STAI_SUCCESS) || (n != 1U))
    {
        return -5;
    }
    voc_out = (float *)out[0];
    if ((flow_in == NULL) || (flow_mask == NULL) || (flow_out == NULL) || (voc_in == NULL) || (voc_out == NULL))
    {
        return -6; /* inputs not preallocated: give them buffers with stai_*_set_inputs() */
    }
    ready = 1;
    return 0;
}

int xl_voice_net_ready(void)
{
    return ready;
}

xl_voice_tensor_t xl_voice_flow_latents(void)
{
    const xl_voice_tensor_t t = {flow_in, XL_STRIDES(STAI_FLOW_IN_1_FLAGS, XL_VOICE_FLOW_FRAMES)};
    return t;
}

float *xl_voice_flow_mask(void)
{
    return flow_mask;
}

int xl_voice_flow_run(xl_voice_tensor_t *out)
{
    const xl_voice_tensor_t t = {flow_out, XL_STRIDES(STAI_FLOW_OUT_1_FLAGS, XL_VOICE_FLOW_FRAMES)};

    if (!ready)
    {
        return -1;
    }
    /* written through the D-cache; the generated code invalidates the outputs itself */
    LL_ATON_Cache_MCU_Clean_Invalidate_Range((uintptr_t)flow_in, STAI_FLOW_IN_1_SIZE_BYTES);
    LL_ATON_Cache_MCU_Clean_Invalidate_Range((uintptr_t)flow_mask, STAI_FLOW_IN_2_SIZE_BYTES);
    if (stai_flow_run(flow_context, STAI_MODE_SYNC) != STAI_SUCCESS)
    {
        return -2;
    }
    *out = t;
    return 0;
}

xl_voice_tensor_t xl_voice_vocoder_latents(void)
{
    const xl_voice_tensor_t t = {voc_in, XL_STRIDES(STAI_VOCODER_IN_1_FLAGS, XL_VOICE_VOC_IN_FRAMES)};
    return t;
}

int xl_voice_vocoder_run(const float **audio)
{
    if (!ready)
    {
        return -1;
    }
    LL_ATON_Cache_MCU_Clean_Invalidate_Range((uintptr_t)voc_in, STAI_VOCODER_IN_1_SIZE_BYTES);
    if (stai_vocoder_run(vocoder_context, STAI_MODE_SYNC) != STAI_SUCCESS)
    {
        return -2;
    }
    *audio = voc_out;
    return 0;
}

#else /* no generated voice networks: text only */

int xl_voice_net_init(void)
{
    return -1;
}

int xl_voice_net_ready(void)
{
    return 0;
}

xl_voice_tensor_t xl_voice_flow_latents(void)
{
    const xl_voice_tensor_t t = {NULL, 0U, 0U};
    return t;
}

float *xl_voice_flow_mask(void)
{
    return NULL;
}

int xl_voice_flow_run(xl_voice_tensor_t *out)
{
    (void)out;
    return -1;
}

xl_voice_tensor_t xl_voice_vocoder_latents(void)
{
    const xl_voice_tensor_t t = {NULL, 0U, 0U};
    return t;
}

int xl_voice_vocoder_run(const float **audio)
{
    (void)audio;
    return -1;
}

#endif
