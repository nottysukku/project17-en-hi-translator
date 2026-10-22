/**
 * @file     tts_front.c
 * @brief    MMS-TTS Hindi front end on the CPU; implements tts_front.h.
 * @details  Mirrors model-tools/tts_front_ref.py step by step (checked against it and against
 *           transformers' VitsModel by model-tools/tts_front_check.py).
 *
 *           Numerics: the text encoder's matrices are int8 in groups of 64 with a float16 scale
 *           per group; before each of them the input rows are quantised to int8 with one scale per
 *           group of 64 (as in it2.c), so each group's dot product is exact integer arithmetic.
 *           Convolutions over time are the same products over three neighbouring rows. The
 *           duration predictor runs in float32 from float16 weights. Layer norms, attention,
 *           softmax and the spline flows are float32.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tts_front.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#define TTS_USE_MVE 1
#else
#define TTS_USE_MVE 0
#endif

/* ------------------------------------------------------------------------------------------ */
/*                                        Constants                                           */
/* ------------------------------------------------------------------------------------------ */

#define D       192U /* hidden width */
#define NH      2U   /* attention heads */
#define HD      96U  /* head width */
#define FF      768U /* feed-forward width */
#define LAYERS  6U
#define WIN     4U   /* relative attention window: offsets -4..4 */
#define GROUP   64U  /* int8 group size */
#define BINS    10U  /* spline bins */
#define PROJ    29U  /* conv flow projection: 10 widths, 10 heights, 9 derivatives */
#define TAIL    5.0f /* spline range [-TAIL, TAIL] */
#define LN_EPS  1e-5f
#define ROWS    8U   /* rows per tile: a tile of int8 inputs stays in the data cache */
#define N       TTS_MAX_IDS

#define KIND_F32 0U
#define KIND_F16 1U
#define KIND_Q8  3U

/* Tensor table (tts_front_pack.py, TENSOR ORDER). */
#define T_EMBED      0U
#define T_LAYER(i)   (1U + ((i) * 18U))
#define T_PROJECT    109U
#define T_SDP_PRE    111U
#define T_SDP_DDS    113U
#define T_SDP_PROJ   137U
#define T_FLOW(k)    (139U + ((k) * 28U)) /* conv flows in reverse order: k = 0, 1, 2 */
#define T_AFFINE     223U
#define T_COUNT      225U
/* Within a layer (T_LAYER(i) + ...) */
#define L_Q      0U
#define L_K      2U
#define L_V      4U
#define L_O      6U
#define L_RELK   8U
#define L_RELV   9U
#define L_LN1    10U
#define L_FFN1   12U
#define L_FFN2   14U
#define L_LN2    16U
/* Within a conv flow (T_FLOW(k) + ...) */
#define F_PRE    0U
#define F_DDS    2U
#define F_PROJ   26U

/* ------------------------------------------------------------------------------------------ */
/*                                        Workspace                                           */
/* ------------------------------------------------------------------------------------------ */

/** @brief Scratch in the caller's workspace. Buffers are reused between the two networks. */
typedef struct
{
    float *x;     /* [N][D] residual stream; afterwards the text encoder output */
    float *q;     /* [N][D] */
    float *k;     /* [N][D] */
    float *v;     /* [N][D] */
    float *h;     /* [N][D] */
    float *stats; /* [N][FF]: feed-forward hidden; afterwards [N][2D] prior means, log-variances */
    int8_t *xq;   /* [N][FF] int8 copy of a matrix product's input rows */
    float *xs;    /* [N][FF / GROUP] their scales */
    float *att;   /* [N] one attention row */
    float *z0;    /* [N] duration flow channel 0 */
    float *z1;    /* [N] duration flow channel 1 */
    float *logw;  /* [N] */
    uint16_t *dur;  /* [N] frames per id */
    uint32_t *start; /* [N + 1] first frame of each id */
} ws_t;

static size_t align32(size_t n)
{
    return (n + 31U) & ~(size_t)31U;
}

size_t tts_front_workspace_size(void)
{
    return align32(sizeof(ws_t)) + (5U * align32(N * D * sizeof(float))) + align32(N * FF * sizeof(float)) +
           align32(N * FF) + align32(N * (FF / GROUP) * sizeof(float)) + (4U * align32(N * sizeof(float))) +
           align32(N * sizeof(uint16_t)) + align32((N + 1U) * sizeof(uint32_t));
}

static ws_t *ws(const tts_front_t *f)
{
    return (ws_t *)f->ws;
}

/* ------------------------------------------------------------------------------------------ */
/*                                     Tensors and numbers                                    */
/* ------------------------------------------------------------------------------------------ */

static float half_to_float(uint16_t h)
{
#if defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 v;
    memcpy(&v, &h, sizeof(v));
    return (float)v;
#else
    const uint32_t sign = (uint32_t)(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t man = h & 0x3FFU;
    uint32_t bits;
    float out;

    if (exp == 0U)
    {
        if (man == 0U)
        {
            bits = sign;
        }
        else
        {
            exp = 127U - 15U + 1U;
            while ((man & 0x400U) == 0U)
            {
                man <<= 1;
                exp--;
            }
            bits = sign | (exp << 23) | ((man & 0x3FFU) << 13);
        }
    }
    else if (exp == 31U)
    {
        bits = sign | 0x7F800000U | (man << 13);
    }
    else
    {
        bits = sign | ((exp + 112U) << 23) | (man << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
#endif
}

static const float *t_f32(const tts_front_t *f, uint32_t i)
{
    return (const float *)(const void *)(f->blob + f->t[i].data_off);
}

static const uint16_t *t_f16(const tts_front_t *f, uint32_t i)
{
    return (const uint16_t *)(const void *)(f->blob + f->t[i].data_off);
}

static void layer_norm(const float *w, const float *b, float *x, uint32_t rows)
{
    uint32_t r, c;

    for (r = 0; r < rows; ++r)
    {
        float *xr = x + ((size_t)r * D);
        float mean = 0.0f, var = 0.0f, inv;

        for (c = 0; c < D; ++c)
        {
            mean += xr[c];
        }
        mean /= (float)D;
        for (c = 0; c < D; ++c)
        {
            const float d = xr[c] - mean;
            var += d * d;
        }
        inv = 1.0f / sqrtf((var / (float)D) + LN_EPS);
        for (c = 0; c < D; ++c)
        {
            xr[c] = ((xr[c] - mean) * inv * w[c]) + b[c];
        }
    }
}

/** @brief Layer norm with float16 weight and bias (duration predictor). */
static void layer_norm16(const tts_front_t *f, uint32_t wi, float *x, uint32_t rows)
{
    float w[D], b[D];
    uint32_t c;

    for (c = 0; c < D; ++c)
    {
        w[c] = half_to_float(t_f16(f, wi)[c]);
        b[c] = half_to_float(t_f16(f, wi + 1U)[c]);
    }
    layer_norm(w, b, x, rows);
}

static float gelu(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

static float softplus(float x)
{
    return (x > 20.0f) ? x : log1pf(expf(x));
}

/* ------------------------------------------------------------------------------------------ */
/*                                 int8 matrix products (Q8)                                  */
/* ------------------------------------------------------------------------------------------ */

/** @brief Rows of x to int8, one scale per group of 64 (scale = max|x| / 127). */
static void quantize_rows(const float *x, uint32_t rows, uint32_t cols, int8_t *q, float *scales)
{
    const uint32_t groups = cols / GROUP;
    uint32_t r, g, c;

    for (r = 0; r < rows; ++r)
    {
        for (g = 0; g < groups; ++g)
        {
            const float *xg = x + ((size_t)r * cols) + (g * GROUP);
            int8_t *qg = q + ((size_t)r * cols) + (g * GROUP);
            float mx = 0.0f, inv, sc;

            for (c = 0; c < GROUP; ++c)
            {
                const float a = fabsf(xg[c]);
                if (a > mx)
                {
                    mx = a;
                }
            }
            sc = (mx > 0.0f) ? (mx / 127.0f) : 1.0f;
            scales[(r * groups) + g] = sc;
            inv = 1.0f / sc;
            for (c = 0; c < GROUP; ++c)
            {
                const float v = xg[c] * inv;
                qg[c] = (int8_t)((v >= 0.0f) ? (v + 0.5f) : (v - 0.5f));
            }
        }
    }
}

#ifndef TTS_FRONT_FLOAT_ACTIVATIONS
/** @brief Dot product of 64 int8 weights with 64 int8 activations. */
static int32_t dot64(const int8_t *w, const int8_t *x)
{
#if TTS_USE_MVE
    int32_t acc = 0;
    acc = vmladavaq_s8(acc, vldrbq_s8(w), vldrbq_s8(x));
    acc = vmladavaq_s8(acc, vldrbq_s8(w + 16), vldrbq_s8(x + 16));
    acc = vmladavaq_s8(acc, vldrbq_s8(w + 32), vldrbq_s8(x + 32));
    acc = vmladavaq_s8(acc, vldrbq_s8(w + 48), vldrbq_s8(x + 48));
    return acc;
#else
    int32_t acc = 0;
    uint32_t i;

    for (i = 0; i < GROUP; ++i)
    {
        acc += (int32_t)w[i] * x[i];
    }
    return acc;
#endif
}
#endif

/**
 * @brief y[r][n] = b[n] + sum over taps j, channels c of W[n][j * C + c] * x[r + j - pad][c],
 *        pad = (taps - 1) / 2, rows outside [0, rows) being zero: a "same" Conv1d over time
 *        (taps 3) or a linear layer (taps 1). W is Q8 [N][taps * C], tap-major.
 */
static void qconv(const tts_front_t *f, uint32_t wi, const float *x, uint32_t rows, uint32_t C, uint32_t taps,
                  float *y)
{
    ws_t *s = ws(f);
    const tts_tensor_t *t = &f->t[wi];
    const int8_t *data = (const int8_t *)(const void *)(f->blob + t->data_off);
    const uint16_t *sc = (const uint16_t *)(const void *)(f->blob + t->scale_off);
    const float *bias = t_f32(f, wi + 1U);
    const uint32_t outs = t->rows, K = t->cols, cg = C / GROUP, kg = K / GROUP, pad = (taps - 1U) / 2U;
    uint32_t r0, n, r, j, g;

    quantize_rows(x, rows, C, s->xq, s->xs);
    for (r0 = 0; r0 < rows; r0 += ROWS)
    {
        const uint32_t r1 = ((r0 + ROWS) < rows) ? (r0 + ROWS) : rows;

        for (n = 0; n < outs; ++n)
        {
            const int8_t *wrow = data + ((size_t)n * K);
            const uint16_t *srow = sc + ((size_t)n * kg);

            for (r = r0; r < r1; ++r)
            {
                float acc = 0.0f;

                for (j = 0; j < taps; ++j)
                {
                    const int32_t src = (int32_t)r + (int32_t)j - (int32_t)pad;
                    const int8_t *xr = NULL;
                    const float *xsr = NULL;

                    if ((src < 0) || (src >= (int32_t)rows))
                    {
                        continue;
                    }
#ifdef TTS_FRONT_FLOAT_ACTIVATIONS
                    /* test build: float inputs times the int8 weights, to check the structure */
                    (void)xr;
                    (void)xsr;
                    for (g = 0; g < cg; ++g)
                    {
                        const float *xf = x + ((size_t)src * C) + (g * GROUP);
                        const int8_t *wg = wrow + (j * C) + (g * GROUP);
                        const float ws = half_to_float(srow[(j * cg) + g]);
                        uint32_t e;
                        for (e = 0; e < GROUP; ++e)
                        {
                            acc += ((float)wg[e] * ws) * xf[e];
                        }
                    }
#else
                    xr = s->xq + ((size_t)src * C);
                    xsr = s->xs + ((size_t)src * cg);
                    for (g = 0; g < cg; ++g)
                    {
                        acc += (float)dot64(wrow + (j * C) + (g * GROUP), xr + (g * GROUP)) *
                               (half_to_float(srow[(j * cg) + g]) * xsr[g]);
                    }
#endif
                }
                y[((size_t)r * outs) + n] = acc + bias[n];
            }
        }
    }
}

/** @brief 1x1 convolution with float16 weights [outs][D] and bias: y[r][o] for r < rows. */
static void pw16(const tts_front_t *f, uint32_t wi, const float *x, uint32_t rows, float *y)
{
    const uint32_t outs = f->t[wi].rows;
    const uint16_t *w = t_f16(f, wi);
    const uint16_t *b = t_f16(f, wi + 1U);
    float wrow[D];
    uint32_t r0, o, r, c;

    for (r0 = 0; r0 < rows; r0 += ROWS)
    {
        const uint32_t r1 = ((r0 + ROWS) < rows) ? (r0 + ROWS) : rows;

        for (o = 0; o < outs; ++o)
        {
            const float bo = half_to_float(b[o]);

            for (c = 0; c < D; ++c)
            {
                wrow[c] = half_to_float(w[((size_t)o * D) + c]);
            }
            for (r = r0; r < r1; ++r)
            {
                const float *xr = x + ((size_t)r * D);
                float acc = 0.0f;
                for (c = 0; c < D; ++c)
                {
                    acc += wrow[c] * xr[c];
                }
                y[((size_t)r * outs) + o] = acc + bo;
            }
        }
    }
}

/* ------------------------------------------------------------------------------------------ */
/*                                       Text encoder                                         */
/* ------------------------------------------------------------------------------------------ */

/** @brief Two-head self-attention with relative positions (-WIN..WIN) on keys and values. */
static void attention(const tts_front_t *f, uint32_t layer, uint32_t n)
{
    ws_t *s = ws(f);
    const float *relk = t_f32(f, T_LAYER(layer) + L_RELK); /* [2 WIN + 1][HD] */
    const float *relv = t_f32(f, T_LAYER(layer) + L_RELV);
    uint32_t hh, i, j, c;

    for (hh = 0; hh < NH; ++hh)
    {
        for (i = 0; i < n; ++i)
        {
            const float *qi = s->q + ((size_t)i * D) + (hh * HD);
            float *oi = s->h + ((size_t)i * D) + (hh * HD);
            float mx = -1e30f, sum = 0.0f;

            for (j = 0; j < n; ++j)
            {
                const float *kj = s->k + ((size_t)j * D) + (hh * HD);
                float d = 0.0f;
                for (c = 0; c < HD; ++c)
                {
                    d += qi[c] * kj[c];
                }
                if (((j + WIN) >= i) && (j <= (i + WIN)))
                {
                    const float *rk = relk + ((size_t)(j + WIN - i) * HD);
                    for (c = 0; c < HD; ++c)
                    {
                        d += qi[c] * rk[c];
                    }
                }
                s->att[j] = d;
                if (d > mx)
                {
                    mx = d;
                }
            }
            for (j = 0; j < n; ++j)
            {
                s->att[j] = expf(s->att[j] - mx);
                sum += s->att[j];
            }
            for (c = 0; c < HD; ++c)
            {
                oi[c] = 0.0f;
            }
            for (j = 0; j < n; ++j)
            {
                const float a = s->att[j] / sum;
                const float *vj = s->v + ((size_t)j * D) + (hh * HD);
                for (c = 0; c < HD; ++c)
                {
                    oi[c] += a * vj[c];
                }
                if (((j + WIN) >= i) && (j <= (i + WIN)))
                {
                    const float *rv = relv + ((size_t)(j + WIN - i) * HD);
                    for (c = 0; c < HD; ++c)
                    {
                        oi[c] += a * rv[c];
                    }
                }
            }
        }
    }
}

static void text_encoder(const tts_front_t *f, const uint16_t *ids, uint32_t n)
{
    ws_t *s = ws(f);
    const float *emb = t_f32(f, T_EMBED);
    const float scale = sqrtf((float)D), qscale = 1.0f / sqrtf((float)HD);
    uint32_t l, i, c;

    for (i = 0; i < n; ++i)
    {
        for (c = 0; c < D; ++c)
        {
            s->x[((size_t)i * D) + c] = emb[((size_t)ids[i] * D) + c] * scale;
        }
    }
    for (l = 0; l < LAYERS; ++l)
    {
        const uint32_t L = T_LAYER(l);

        qconv(f, L + L_Q, s->x, n, D, 1U, s->q);
        qconv(f, L + L_K, s->x, n, D, 1U, s->k);
        qconv(f, L + L_V, s->x, n, D, 1U, s->v);
        for (i = 0; i < n * D; ++i)
        {
            s->q[i] *= qscale;
        }
        attention(f, l, n);
        qconv(f, L + L_O, s->h, n, D, 1U, s->q);
        for (i = 0; i < n * D; ++i)
        {
            s->x[i] += s->q[i];
        }
        layer_norm(t_f32(f, L + L_LN1), t_f32(f, L + L_LN1 + 1U), s->x, n);
        qconv(f, L + L_FFN1, s->x, n, D, 3U, s->stats);
        for (i = 0; i < n * FF; ++i)
        {
            if (s->stats[i] < 0.0f)
            {
                s->stats[i] = 0.0f;
            }
        }
        qconv(f, L + L_FFN2, s->stats, n, FF, 3U, s->q);
        for (i = 0; i < n * D; ++i)
        {
            s->x[i] += s->q[i];
        }
        layer_norm(t_f32(f, L + L_LN2), t_f32(f, L + L_LN2 + 1U), s->x, n);
    }
    qconv(f, T_PROJECT, s->x, n, D, 1U, s->stats); /* [n][2D]: means, then log-variances */
}

/* ------------------------------------------------------------------------------------------ */
/*                              Stochastic duration predictor                                 */
/* ------------------------------------------------------------------------------------------ */

/**
 * @brief Dilated depth-separable convolutions (3 layers, dilations 1, 3, 9) on x [n][D], in place;
 *        g (if not NULL) is added first. Uses s->v and s->h as scratch.
 */
static void dds(const tts_front_t *f, uint32_t wi, float *x, const float *g, uint32_t n)
{
    ws_t *s = ws(f);
    uint32_t i, t, c;

    if (g != NULL)
    {
        for (i = 0; i < n * D; ++i)
        {
            x[i] += g[i];
        }
    }
    for (i = 0; i < 3U; ++i)
    {
        const uint32_t base = wi + (i * 8U), d = (i == 0U) ? 1U : ((i == 1U) ? 3U : 9U);
        const uint16_t *w = t_f16(f, base); /* [D][3] */
        const uint16_t *b = t_f16(f, base + 1U);

        for (t = 0; t < n; ++t)
        {
            for (c = 0; c < D; ++c)
            {
                float acc = half_to_float(b[c]) + (half_to_float(w[(c * 3U) + 1U]) * x[((size_t)t * D) + c]);
                if (t >= d)
                {
                    acc += half_to_float(w[c * 3U]) * x[((size_t)(t - d) * D) + c];
                }
                if ((t + d) < n)
                {
                    acc += half_to_float(w[(c * 3U) + 2U]) * x[((size_t)(t + d) * D) + c];
                }
                s->v[((size_t)t * D) + c] = acc;
            }
        }
        layer_norm16(f, base + 2U, s->v, n);
        for (t = 0; t < n * D; ++t)
        {
            s->v[t] = gelu(s->v[t]);
        }
        pw16(f, base + 4U, s->v, n, s->h);
        layer_norm16(f, base + 6U, s->h, n);
        for (t = 0; t < n * D; ++t)
        {
            x[t] += gelu(s->h[t]);
        }
    }
}

/** @brief Inverse of the rational-quadratic spline on [-TAIL, TAIL], identity outside. */
static float spline_inverse(float y, const float *uw, const float *uh, const float *ud)
{
    const float minw = 1e-3f, minh = 1e-3f, mind = 1e-3f;
    float w[BINS], hgt[BINS], cw[BINS + 1U], ch[BINS + 1U], d[BINS + 1U];
    float mx, sum, delta, i1, i2, i3, a, b, c, disc, root;
    uint32_t k, bin;

    if ((y < -TAIL) || (y > TAIL))
    {
        return y;
    }
    /* widths and heights: softmax, a minimum share per bin, cumulative, scaled to the range */
    for (bin = 0; bin < 2U; ++bin)
    {
        const float *u = (bin == 0U) ? uw : uh;
        float *cum = (bin == 0U) ? cw : ch;
        float *size = (bin == 0U) ? w : hgt;
        const float minv = (bin == 0U) ? minw : minh;

        mx = u[0];
        for (k = 1; k < BINS; ++k)
        {
            mx = (u[k] > mx) ? u[k] : mx;
        }
        sum = 0.0f;
        for (k = 0; k < BINS; ++k)
        {
            size[k] = expf(u[k] - mx);
            sum += size[k];
        }
        cum[0] = 0.0f;
        for (k = 0; k < BINS; ++k)
        {
            cum[k + 1U] = cum[k] + (minv + ((1.0f - (minv * (float)BINS)) * (size[k] / sum)));
        }
        for (k = 0; k <= BINS; ++k)
        {
            cum[k] = ((2.0f * TAIL) * cum[k]) - TAIL;
        }
        cum[0] = -TAIL;
        cum[BINS] = TAIL;
        for (k = 0; k < BINS; ++k)
        {
            size[k] = cum[k + 1U] - cum[k];
        }
    }
    d[0] = mind + softplus(logf(expf(1.0f - mind) - 1.0f));
    d[BINS] = d[0];
    for (k = 1; k < BINS; ++k)
    {
        d[k] = mind + softplus(ud[k - 1U]);
    }
    /* bin: the last height edge at or below y (the top edge widened by 1e-6) */
    bin = 0;
    for (k = 0; k <= BINS; ++k)
    {
        const float edge = (k == BINS) ? (ch[BINS] + 1e-6f) : ch[k];
        if (y >= edge)
        {
            bin = k;
        }
    }
    if (bin >= BINS)
    {
        bin = BINS - 1U;
    }
    delta = hgt[bin] / w[bin];
    i1 = d[bin] + d[bin + 1U] - (2.0f * delta);
    i2 = y - ch[bin];
    i3 = i2 * i1;
    a = (hgt[bin] * (delta - d[bin])) + i3;
    b = (hgt[bin] * d[bin]) - i3;
    c = -delta * i2;
    disc = (b * b) - (4.0f * a * c);
    if (disc < 0.0f)
    {
        disc = 0.0f;
    }
    root = (2.0f * c) / (-b - sqrtf(disc));
    return (root * w[bin]) + cw[bin];
}

static void durations(tts_front_t *f, uint32_t n)
{
    ws_t *s = ws(f);
    const float inv_sqrt = 1.0f / sqrtf((float)D);
    const float *tr = t_f32(f, T_AFFINE), *ls = t_f32(f, T_AFFINE + 1U);
    float *g = s->k; /* conditioning [n][D], kept for the three flows */
    uint32_t k, t, c;

    pw16(f, T_SDP_PRE, s->x, n, s->q);
    dds(f, T_SDP_DDS, s->q, NULL, n);
    pw16(f, T_SDP_PROJ, s->q, n, g);
    for (t = 0; t < n; ++t)
    {
        s->z0[t] = tts_front_gauss(f->seed, 0U, t) * f->hdr->noise_scale_duration;
        s->z1[t] = tts_front_gauss(f->seed, 0U, N + t) * f->hdr->noise_scale_duration;
    }
    for (k = 0; k < 3U; ++k)
    {
        const uint32_t F = T_FLOW(k);
        const uint16_t *pw = t_f16(f, F + F_PRE), *pb = t_f16(f, F + F_PRE + 1U);
        float *tmp = s->z0;

        s->z0 = s->z1; /* the flip before every flow swaps the two channels */
        s->z1 = tmp;
        for (t = 0; t < n; ++t)
        {
            for (c = 0; c < D; ++c)
            {
                s->q[((size_t)t * D) + c] = (half_to_float(pw[c]) * s->z0[t]) + half_to_float(pb[c]);
            }
        }
        dds(f, F + F_DDS, s->q, g, n);
        pw16(f, F + F_PROJ, s->q, n, s->stats + ((size_t)n * 2U * D)); /* [n][PROJ] after the stats */
        for (t = 0; t < n; ++t)
        {
            const float *p = s->stats + ((size_t)n * 2U * D) + ((size_t)t * PROJ);
            float uw[BINS], uh[BINS];
            uint32_t b;
            for (b = 0; b < BINS; ++b)
            {
                uw[b] = p[b] * inv_sqrt;
                uh[b] = p[BINS + b] * inv_sqrt;
            }
            s->z1[t] = spline_inverse(s->z1[t], uw, uh, p + (2U * BINS));
        }
    }
    {
        float *tmp = s->z0; /* last flip, then the elementwise affine flow */
        s->z0 = s->z1;
        s->z1 = tmp;
    }
    for (t = 0; t < n; ++t)
    {
        s->logw[t] = (s->z0[t] - tr[0]) * expf(-ls[0]);
    }
}

/* ------------------------------------------------------------------------------------------ */
/*                                          Noise                                             */
/* ------------------------------------------------------------------------------------------ */

static uint32_t mix32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x21F0AAADU;
    x ^= x >> 15;
    x *= 0x735A2D97U;
    x ^= x >> 15;
    return x;
}

float tts_front_gauss(uint32_t seed, uint32_t stream, uint32_t index)
{
    const uint32_t key = mix32(seed ^ (stream * 0x9E3779B9U));
    const uint32_t a = mix32(key ^ (2U * index));
    const uint32_t b = mix32(key ^ ((2U * index) + 1U));
    const float u1 = ((float)(a >> 8) + 1.0f) * (1.0f / 16777216.0f); /* (0, 1] */
    const float u2 = (float)(b >> 8) * (1.0f / 16777216.0f);          /* [0, 1) */

    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318531f * u2);
}

/* ------------------------------------------------------------------------------------------ */
/*                                         Text                                               */
/* ------------------------------------------------------------------------------------------ */

static uint32_t utf8_next(const uint8_t **p)
{
    const uint8_t *s = *p;
    uint32_t c;

    if (s[0] < 0x80U)
    {
        c = s[0];
        *p = s + 1;
    }
    else if (((s[0] & 0xE0U) == 0xC0U) && (s[1] != 0U))
    {
        c = ((uint32_t)(s[0] & 0x1FU) << 6) | (s[1] & 0x3FU);
        *p = s + 2;
    }
    else if (((s[0] & 0xF0U) == 0xE0U) && (s[1] != 0U) && (s[2] != 0U))
    {
        c = ((uint32_t)(s[0] & 0x0FU) << 12) | ((uint32_t)(s[1] & 0x3FU) << 6) | (s[2] & 0x3FU);
        *p = s + 3;
    }
    else if (((s[0] & 0xF8U) == 0xF0U) && (s[1] != 0U) && (s[2] != 0U) && (s[3] != 0U))
    {
        c = ((uint32_t)(s[0] & 0x07U) << 18) | ((uint32_t)(s[1] & 0x3FU) << 12) | ((uint32_t)(s[2] & 0x3FU) << 6) |
            (s[3] & 0x3FU);
        *p = s + 4;
    }
    else
    {
        c = 0xFFFDU; /* invalid byte: skipped as an unknown character */
        *p = s + 1;
    }
    return c;
}

/** @brief Id of a code point in the model's alphabet, or -1. */
static int32_t symbol_id(const tts_front_t *f, uint32_t cp)
{
    const uint32_t *v = (const uint32_t *)(const void *)(f->blob + f->hdr->vocab_off);
    uint32_t i;

    for (i = 0; i < f->hdr->n_vocab; ++i)
    {
        if (v[2U * i] == cp)
        {
            return (int32_t)v[(2U * i) + 1U];
        }
    }
    return -1;
}

/** @brief The code points one input character stands for (at most 2); returns how many. */
static uint32_t spell(uint32_t cp, uint32_t out[2])
{
    static const uint16_t nukta_base[8] = {0x0915U, 0x0916U, 0x0917U, 0x091CU, 0x0921U, 0x0922U, 0x092BU, 0x092FU};

    if ((cp >= 0x0958U) && (cp <= 0x095FU))
    {
        out[0] = nukta_base[cp - 0x0958U];
        out[1] = 0x093CU;
        return 2;
    }
    switch (cp)
    {
    case 0x0929U: /* NNNA */
        out[0] = 0x0928U;
        out[1] = 0x093CU;
        return 2;
    case 0x0931U: /* RRA */
        out[0] = 0x0930U;
        out[1] = 0x093CU;
        return 2;
    case 0x0934U: /* LLLA */
        out[0] = 0x0933U;
        out[1] = 0x093CU;
        return 2;
    case 0x090BU: /* vocalic R: ri */
        out[0] = 0x0930U;
        out[1] = 0x093FU;
        return 2;
    case 0x0960U: /* vocalic RR: rii */
        out[0] = 0x0930U;
        out[1] = 0x0940U;
        return 2;
    case 0x0944U: /* vowel sign vocalic RR */
        out[0] = 0x0943U;
        return 1;
    default:
        out[0] = cp;
        return 1;
    }
}

uint32_t tts_front_text_to_ids(const tts_front_t *f, const char *utf8, uint16_t *ids, uint32_t max)
{
    const uint8_t *p = (const uint8_t *)utf8;
    const int32_t space = symbol_id(f, 0x20U);
    uint32_t n = 0, last_char = 0;

    if ((f == NULL) || (utf8 == NULL) || (max < 3U))
    {
        return 0;
    }
    ids[n++] = 0;
    while ((*p != 0U) && ((n + 2U) <= max))
    {
        uint32_t cps[2], k, count = spell(utf8_next(&p), cps);

        for (k = 0; (k < count) && ((n + 2U) <= max); ++k)
        {
            const int32_t id = symbol_id(f, cps[k]);
            if ((id < 0) || ((id == space) && (n == 1U)))
            {
                continue; /* not in the alphabet, or a leading space */
            }
            ids[n++] = (uint16_t)id;
            ids[n++] = 0;
            if (id != space)
            {
                last_char = n;
            }
        }
    }
    /* trailing spaces go, like Python's strip() */
    return (last_char > 0U) ? last_char : 0U;
}

/* ------------------------------------------------------------------------------------------ */
/*                                       Public API                                           */
/* ------------------------------------------------------------------------------------------ */

int tts_front_open(tts_front_t *f, const void *blob, void *workspace)
{
    const tts_header_t *h = (const tts_header_t *)blob;
    uint8_t *w = (uint8_t *)workspace;
    ws_t *s;

    if ((f == NULL) || (blob == NULL) || (workspace == NULL) || (memcmp(h->magic, "TTSF", 4) != 0) ||
        (h->version != 1U) || (h->hidden != D) || (h->heads != NH) || (h->ffn != FF) || (h->layers != LAYERS) ||
        (h->window != WIN) || (h->n_tensors != T_COUNT))
    {
        return -1;
    }
    memset(f, 0, sizeof(*f));
    f->blob = (const uint8_t *)blob;
    f->hdr = h;
    f->t = (const tts_tensor_t *)(const void *)(f->blob + h->table_off);
    if ((f->t[T_LAYER(0) + L_FFN1].kind != KIND_Q8) || (f->t[T_LAYER(0) + L_FFN1].cols != 3U * D) ||
        (f->t[T_PROJECT].rows != 2U * D) || (f->t[T_FLOW(2) + F_PROJ].rows != PROJ) ||
        (f->t[T_SDP_PRE].kind != KIND_F16) || (f->t[T_AFFINE].kind != KIND_F32))
    {
        return -2;
    }
    f->ws = workspace;
    s = (ws_t *)(void *)w;
    w += align32(sizeof(ws_t));
#define TAKE(ptr, type, bytes)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        (ptr) = (type *)(void *)w;                                                                                     \
        w += align32(bytes);                                                                                           \
    } while (0)
    TAKE(s->x, float, N * D * sizeof(float));
    TAKE(s->q, float, N * D * sizeof(float));
    TAKE(s->k, float, N * D * sizeof(float));
    TAKE(s->v, float, N * D * sizeof(float));
    TAKE(s->h, float, N * D * sizeof(float));
    TAKE(s->stats, float, N * FF * sizeof(float));
    TAKE(s->xq, int8_t, N * FF);
    TAKE(s->xs, float, N * (FF / GROUP) * sizeof(float));
    TAKE(s->att, float, N * sizeof(float));
    TAKE(s->z0, float, N * sizeof(float));
    TAKE(s->z1, float, N * sizeof(float));
    TAKE(s->logw, float, N * sizeof(float));
    TAKE(s->dur, uint16_t, N * sizeof(uint16_t));
    TAKE(s->start, uint32_t, (N + 1U) * sizeof(uint32_t));
#undef TAKE
    return 0;
}

uint32_t tts_front_run(tts_front_t *f, const uint16_t *ids, uint32_t n, uint32_t seed)
{
    ws_t *s;
    uint32_t i;

    if ((f == NULL) || (f->ws == NULL) || (ids == NULL) || (n == 0U) || (n > N))
    {
        return 0;
    }
    for (i = 0; i < n; ++i)
    {
        if (ids[i] >= f->hdr->n_vocab)
        {
            return 0;
        }
    }
    s = ws(f);
    f->n_ids = n;
    f->seed = seed;
    text_encoder(f, ids, n);
    durations(f, n);
    s->start[0] = 0;
    for (i = 0; i < n; ++i)
    {
        float frames = ceilf(expf(s->logw[i]) * f->hdr->length_scale);
        if (!(frames >= 0.0f))
        {
            frames = 0.0f; /* NaN guard */
        }
        if (frames > 1000.0f)
        {
            frames = 1000.0f;
        }
        s->dur[i] = (uint16_t)frames;
        s->start[i + 1U] = s->start[i] + s->dur[i];
    }
    f->n_frames = (s->start[n] > 0U) ? s->start[n] : 1U;
    return f->n_frames;
}

const uint16_t *tts_front_durations(const tts_front_t *f)
{
    return ws(f)->dur;
}

const float *tts_front_prior_mean(const tts_front_t *f)
{
    return ws(f)->stats;
}

const float *tts_front_prior_logvar(const tts_front_t *f)
{
    return ws(f)->stats + D;
}

const float *tts_front_logw(const tts_front_t *f)
{
    return ws(f)->logw;
}

void tts_front_latents(const tts_front_t *f, uint32_t start, uint32_t count, float *z, uint32_t c_stride,
                       uint32_t t_stride)
{
    const ws_t *s = ws(f);
    const float ns = f->hdr->noise_scale;
    uint32_t id = 0, i, c;

    /* the id whose frames contain `start` */
    while (((id + 1U) < f->n_ids) && (s->start[id + 1U] <= start))
    {
        id++;
    }
    for (i = 0; i < count; ++i)
    {
        const uint32_t frame = start + i;

        while (((id + 1U) < f->n_ids) && (s->start[id + 1U] <= frame))
        {
            id++;
        }
        for (c = 0; c < TTS_CHANNELS; ++c)
        {
            float v = 0.0f;
            if (frame < s->start[f->n_ids])
            {
                const float *row = s->stats + ((size_t)id * 2U * D);
                v = row[c] + (tts_front_gauss(f->seed, 1U, (frame * TTS_CHANNELS) + c) * expf(row[D + c]) * ns);
            }
            z[((size_t)c * c_stride) + ((size_t)i * t_stride)] = v;
        }
    }
}
