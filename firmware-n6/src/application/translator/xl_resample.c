/**
 * @file     xl_resample.c
 * @brief    3x polyphase interpolation; implements xl_resample.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xl_resample.h"

#include <string.h>

/* model-tools/resample_design.py: phase p, tap k multiplies history[k] (oldest first). */
/* passband ripple 0-7 kHz: 0.01 dB; worst stopband >= 9 kHz: -63.9 dB */
static const float taps[XL_RESAMPLE_PHASES][XL_RESAMPLE_TAPS] = {
    {-0.000259046779f, 0.000570963976f, -0.0010633745f, 0.00179439106f, -0.00283269856f, 0.0042605261f, -0.0061796841f, 0.00872356257f, -0.0120810575f, 0.0165459182f, -0.0226256325f, 0.0313091745f, -0.0448435406f, 0.0696607611f, -0.134354196f, 0.95474775f, 0.189518198f, -0.0835829922f, 0.0512819443f, -0.0351026716f, 0.0251583558f, -0.0183581762f, 0.0134272039f, -0.00974066102f, 0.00695036876f, -0.0048398773f, 0.00326081794f, -0.00210260886f, 0.00127738027f, -0.000712373854f, 0.00034616551f, -0.000126843962f},
    {-0.00037294655f, 0.000899068617f, -0.00175159009f, 0.00304041656f, -0.00489540786f, 0.00747103781f, -0.0109559605f, 0.0155921162f, -0.0217126657f, 0.0298190825f, -0.0407467936f, 0.0560559917f, -0.0790958861f, 0.118624513f, -0.206903778f, 0.634904755f, 0.634904755f, -0.206903778f, 0.118624513f, -0.0790958861f, 0.0560559917f, -0.0407467936f, 0.0298190825f, -0.0217126657f, 0.0155921162f, -0.0109559605f, 0.00747103781f, -0.00489540786f, 0.00304041656f, -0.00175159009f, 0.000899068617f, -0.00037294655f},
    {-0.000126843962f, 0.00034616551f, -0.000712373854f, 0.00127738027f, -0.00210260886f, 0.00326081794f, -0.0048398773f, 0.00695036876f, -0.00974066102f, 0.0134272039f, -0.0183581762f, 0.0251583558f, -0.0351026716f, 0.0512819443f, -0.0835829922f, 0.189518198f, 0.95474775f, -0.134354196f, 0.0696607611f, -0.0448435406f, 0.0313091745f, -0.0226256325f, 0.0165459182f, -0.0120810575f, 0.00872356257f, -0.0061796841f, 0.0042605261f, -0.00283269856f, 0.00179439106f, -0.0010633745f, 0.000570963976f, -0.000259046779f},
};

void xl_resample_reset(xl_resample_t *r)
{
    (void)memset(r, 0, sizeof(*r));
}

void xl_resample_run(xl_resample_t *r, const float *in, uint32_t n, float gain, int16_t *out)
{
    const float scale = gain * 32767.0f;
    uint32_t i, p, k;

    for (i = 0; i < n; ++i)
    {
        (void)memmove(&r->history[0], &r->history[1], (XL_RESAMPLE_TAPS - 1U) * sizeof(float));
        r->history[XL_RESAMPLE_TAPS - 1U] = in[i];
        for (p = 0; p < XL_RESAMPLE_PHASES; ++p)
        {
            float acc = 0.0f;
            int32_t v;

            for (k = 0; k < XL_RESAMPLE_TAPS; ++k)
            {
                acc += taps[p][k] * r->history[k];
            }
            acc *= scale;
            v = (int32_t)((acc >= 0.0f) ? (acc + 0.5f) : (acc - 0.5f));
            if (v > 32767)
            {
                v = 32767;
            }
            else if (v < -32768)
            {
                v = -32768;
            }
            out[(i * XL_RESAMPLE_PHASES) + p] = (int16_t)v;
        }
    }
}
