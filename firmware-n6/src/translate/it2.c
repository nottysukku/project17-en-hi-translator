/**
 * @file     it2.c
 * @brief    IndicTrans2 English-to-Hindi engine; implements it2.h.
 * @details  Numerics: weights are signed 4-bit in groups of 64 with a float16 scale per group
 *           (it2_pack.py). Before each linear layer the input row is quantised to int8 with one
 *           scale per group of 64 (transformer activations have outliers; one scale per row cost
 *           6 chrF points), so a group's dot product is exact integer arithmetic; both scales are
 *           applied in float. Layer norms, attention, GELU and the residual stream are float32.
 *           This matches it2_pack.py's simulation to within float rounding.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "it2.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#define IT2_USE_MVE 1
#else
#define IT2_USE_MVE 0
#endif

/* ------------------------------------------------------------------------------------------ */
/*                                        Constants                                           */
/* ------------------------------------------------------------------------------------------ */

#define D      512U   /* model width */
#define H      8U     /* attention heads */
#define HD     64U    /* head width */
#define FFN    2048U  /* feed-forward width */
#define GROUP  64U    /* Q4 group size */
#define LN_EPS 1e-5f

/* GELU by table over [-GELU_RANGE, GELU_RANGE] with linear interpolation (error < 1e-6). */
#define GELU_RANGE 8.0f
#define GELU_STEPS 4096U

/* Most Hindi words take 1-3 tokens; stop if one token repeats this often in a row. */
#define IT2_MAX_REPEAT 6U

/* ------------------------------------------------------------------------------------------ */
/*                                        Workspace                                           */
/* ------------------------------------------------------------------------------------------ */

/** @brief Scratch carved out of the caller's workspace by it2_open(). */
typedef struct
{
    float *x, *h, *q, *k, *v, *att, *ffn; /* encoder / decoder activations */
    int8_t *xq;                           /* int8 copy of a linear layer's input rows */
    float *xs;                            /* their scales, one per row and group of 64 */
    float *cross_k, *cross_v;             /* [n_dec][IT2_MAX_SRC][D] */
    float *self_k, *self_v;               /* [n_dec][IT2_MAX_TGT][D] */
    float *logits;                        /* [tgt_vocab] */
    float *gelu;                          /* [GELU_STEPS + 1] */
    uint32_t *src_ids;
} it2_scratch_t;

static it2_scratch_t *scratch_of(const it2_model_t *m)
{
    return (it2_scratch_t *)(void *)(m->dec + m->hdr->n_dec);
}

static size_t align32(size_t n)
{
    return (n + 31U) & ~(size_t)31U;
}

size_t it2_workspace_size(const void *blob)
{
    const it2_header_t *h = (const it2_header_t *)blob;
    size_t n = 0;

    n += align32(h->n_enc * sizeof(it2_enc_layer_t));
    n += align32(h->n_dec * sizeof(it2_dec_layer_t));
    n += align32(sizeof(it2_scratch_t));
    n += 5U * align32(IT2_MAX_SRC * D * sizeof(float));             /* x h q k v */
    n += align32(H * IT2_MAX_TGT * sizeof(float));                  /* attention row */
    n += align32(IT2_MAX_SRC * FFN * sizeof(float));                /* ffn */
    n += align32(IT2_MAX_SRC * FFN);                                /* xq */
    n += align32(IT2_MAX_SRC * (FFN / GROUP) * sizeof(float));      /* xs */
    n += 2U * align32((size_t)h->n_dec * IT2_MAX_SRC * D * sizeof(float));
    n += 2U * align32((size_t)h->n_dec * IT2_MAX_TGT * D * sizeof(float));
    n += align32(h->tgt_vocab * sizeof(float));
    n += align32((GELU_STEPS + 1U) * sizeof(float));
    n += align32(IT2_MAX_SRC * sizeof(uint32_t));
    return n;
}

/* ------------------------------------------------------------------------------------------ */
/*                                     Numeric kernels                                        */
/* ------------------------------------------------------------------------------------------ */

static float half_to_float(uint16_t h)
{
#if defined(__ARM_FP16_FORMAT_IEEE)
    __fp16 f;
    memcpy(&f, &h, sizeof(f));
    return (float)f;
#else
    const uint32_t sign = (uint32_t)(h & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t man = h & 0x3FFU;
    uint32_t bits;
    float f;

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
    memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

static const uint8_t *tensor_data(const it2_model_t *m, const it2_tensor_t *t)
{
    return m->blob + t->data_off;
}

static const uint16_t *tensor_scales(const it2_model_t *m, const it2_tensor_t *t)
{
    return (const uint16_t *)(const void *)(m->blob + t->scale_off);
}

/** @brief One Q4 row, dequantised to float (embedding lookups). */
static void q4_row(const it2_model_t *m, const it2_tensor_t *t, uint32_t row, float scale, float *out)
{
    const uint32_t cols = t->cols;
    const uint8_t *data = tensor_data(m, t) + (size_t)row * (cols / 2U);
    const uint16_t *sc = tensor_scales(m, t) + (size_t)row * (cols / GROUP);
    uint32_t g;

    for (g = 0; g < cols / GROUP; ++g)
    {
        const float s = half_to_float(sc[g]) * scale;
        uint32_t blk;

        for (blk = 0; blk < 2U; ++blk)
        {
            const uint8_t *b = data + (g * 32U) + (blk * 16U);
            float *o = out + (g * GROUP) + (blk * 32U);
            uint32_t i;

            for (i = 0; i < 16U; ++i)
            {
                o[i] = (float)((int8_t)(uint8_t)(b[i] << 4) >> 4) * s;
                o[i + 16U] = (float)((int8_t)b[i] >> 4) * s;
            }
        }
    }
}

/** @brief Quantise rows of x to int8 with one scale per group of 64 (scale = max|x| / 127). */
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

/** @brief Integer dot product of one 64-value Q4 group with 64 int8 activations. */
static int32_t q4_group_dot(const uint8_t *b, const int8_t *x)
{
#if IT2_USE_MVE
    int32_t acc = 0;
    int8x16_t v = vldrbq_s8((const int8_t *)b);
    acc = vmladavaq_s8(acc, vshrq_n_s8(vshlq_n_s8(v, 4), 4), vldrbq_s8(x));
    acc = vmladavaq_s8(acc, vshrq_n_s8(v, 4), vldrbq_s8(x + 16));
    v = vldrbq_s8((const int8_t *)(b + 16));
    acc = vmladavaq_s8(acc, vshrq_n_s8(vshlq_n_s8(v, 4), 4), vldrbq_s8(x + 32));
    acc = vmladavaq_s8(acc, vshrq_n_s8(v, 4), vldrbq_s8(x + 48));
    return acc;
#else
    int32_t acc = 0;
    uint32_t blk, i;

    for (blk = 0; blk < 2U; ++blk)
    {
        const uint8_t *bb = b + (blk * 16U);
        const int8_t *xx = x + (blk * 32U);

        for (i = 0; i < 16U; ++i)
        {
            acc += (int32_t)((int8_t)(uint8_t)(bb[i] << 4) >> 4) * xx[i];
            acc += (int32_t)((int8_t)bb[i] >> 4) * xx[i + 16U];
        }
    }
    return acc;
#endif
}

/**
 * @brief y[r][n] = sum_k W[n][k] x[r][k] + b[n] for rows r < rows (row-major x and y).
 * @note  Every weight row is read once for all input rows, so the encoder streams the weights
 *        once per layer and the decoder once per token.
 */
static void linear(const it2_model_t *m, const it2_linear_t *l, const float *x, uint32_t rows, float *y)
{
    it2_scratch_t *s = scratch_of(m);
    const it2_tensor_t *t = l->w;
    const uint32_t K = t->cols;
    const uint32_t N = t->rows;
    const uint32_t groups = K / GROUP;
    const uint8_t *data = tensor_data(m, t);
    const uint16_t *sc = tensor_scales(m, t);
    uint32_t n, r, g;

    quantize_rows(x, rows, K, s->xq, s->xs);
    for (n = 0; n < N; ++n)
    {
        const uint8_t *wrow = data + (size_t)n * (K / 2U);
        const uint16_t *srow = sc + (size_t)n * groups;
        const float bias = (l->b != NULL) ? l->b[n] : 0.0f;

        for (r = 0; r < rows; ++r)
        {
            const int8_t *xr = s->xq + (size_t)r * K;
            const float *xsr = s->xs + (size_t)r * groups;
            float acc = 0.0f;

            for (g = 0; g < groups; ++g)
            {
                acc += (float)q4_group_dot(wrow + (g * 32U), xr + (g * GROUP)) * (half_to_float(srow[g]) * xsr[g]);
            }
            y[(size_t)r * N + n] = acc + bias;
        }
    }
}

static void layer_norm(const it2_norm_t *nrm, const float *x, uint32_t rows, float *y)
{
    uint32_t r, c;

    for (r = 0; r < rows; ++r)
    {
        const float *xr = x + (size_t)r * D;
        float *yr = y + (size_t)r * D;
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
            yr[c] = ((xr[c] - mean) * inv * nrm->w[c]) + nrm->b[c];
        }
    }
}

static void gelu_inplace(const it2_scratch_t *s, float *x, size_t n)
{
    const float step = (2.0f * GELU_RANGE) / (float)GELU_STEPS;
    const float inv_step = 1.0f / step;
    size_t i;

    for (i = 0; i < n; ++i)
    {
        const float v = x[i];
        if (v >= GELU_RANGE)
        {
            continue; /* gelu(x) = x */
        }
        if (v <= -GELU_RANGE)
        {
            x[i] = 0.0f;
            continue;
        }
        {
            const float pos = (v + GELU_RANGE) * inv_step;
            const uint32_t j = (uint32_t)pos;
            const float f = pos - (float)j;
            x[i] = s->gelu[j] + ((s->gelu[j + 1U] - s->gelu[j]) * f);
        }
    }
}

/** @brief Sinusoidal position embedding of position id p (fairseq: ids start at 2), added to x. */
static void add_position(uint32_t p, float *x)
{
    const float step = logf(10000.0f) / (float)((D / 2U) - 1U);
    uint32_t j;

    for (j = 0; j < D / 2U; ++j)
    {
        const float a = (float)p * expf(-(float)j * step);
        x[j] += sinf(a);
        x[j + (D / 2U)] += cosf(a);
    }
}

/**
 * @brief Multi-head attention of nq query rows over nk key/value rows (all row-major, width D).
 * @param causal_offset If >= 0, query i may only see keys <= i + causal_offset.
 */
static void attention(const float *q, uint32_t nq, const float *k, const float *v, uint32_t nk, float *att,
                      float *out)
{
    const float scale = 1.0f / sqrtf((float)HD);
    uint32_t i, h, j, c;

    for (i = 0; i < nq; ++i)
    {
        for (h = 0; h < H; ++h)
        {
            const float *qi = q + (size_t)i * D + (h * HD);
            float *oi = out + (size_t)i * D + (h * HD);
            float mx = -1e30f, sum = 0.0f;

            for (j = 0; j < nk; ++j)
            {
                const float *kj = k + (size_t)j * D + (h * HD);
                float d = 0.0f;
                for (c = 0; c < HD; ++c)
                {
                    d += qi[c] * kj[c];
                }
                att[j] = d * scale;
                if (att[j] > mx)
                {
                    mx = att[j];
                }
            }
            for (j = 0; j < nk; ++j)
            {
                att[j] = expf(att[j] - mx);
                sum += att[j];
            }
            for (c = 0; c < HD; ++c)
            {
                oi[c] = 0.0f;
            }
            for (j = 0; j < nk; ++j)
            {
                const float w = att[j] / sum;
                const float *vj = v + (size_t)j * D + (h * HD);
                for (c = 0; c < HD; ++c)
                {
                    oi[c] += w * vj[c];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------------------------ */
/*                                    Encoder and decoder                                     */
/* ------------------------------------------------------------------------------------------ */

static void encode(const it2_model_t *m, const uint32_t *ids, uint32_t n)
{
    it2_scratch_t *s = scratch_of(m);
    const float emb_scale = sqrtf((float)D);
    uint32_t i, l;

    for (i = 0; i < n; ++i)
    {
        q4_row(m, m->src_emb, ids[i], emb_scale, s->x + (size_t)i * D);
        add_position(i + 2U, s->x + (size_t)i * D);
    }
    layer_norm(&m->enc_emb_norm, s->x, n, s->x);
    for (l = 0; l < m->hdr->n_enc; ++l)
    {
        const it2_enc_layer_t *L = &m->enc[l];

        layer_norm(&L->sa_norm, s->x, n, s->h);
        linear(m, &L->q, s->h, n, s->q);
        linear(m, &L->k, s->h, n, s->k);
        linear(m, &L->v, s->h, n, s->v);
        attention(s->q, n, s->k, s->v, n, s->att, s->h);
        linear(m, &L->o, s->h, n, s->q);
        for (i = 0; i < n * D; ++i)
        {
            s->x[i] += s->q[i];
        }
        layer_norm(&L->ff_norm, s->x, n, s->h);
        linear(m, &L->fc1, s->h, n, s->ffn);
        gelu_inplace(s, s->ffn, (size_t)n * FFN);
        linear(m, &L->fc2, s->ffn, n, s->q);
        for (i = 0; i < n * D; ++i)
        {
            s->x[i] += s->q[i];
        }
    }
    layer_norm(&m->enc_final_norm, s->x, n, s->x);
    /* Cross-attention keys and values of every decoder layer, computed once per sentence. */
    for (l = 0; l < m->hdr->n_dec; ++l)
    {
        linear(m, &m->dec[l].ck, s->x, n, s->cross_k + (size_t)l * IT2_MAX_SRC * D);
        linear(m, &m->dec[l].cv, s->x, n, s->cross_v + (size_t)l * IT2_MAX_SRC * D);
    }
}

/** @brief One decoder step: token at position pos -> logits over the kept Hindi vocabulary. */
static void decode_step(const it2_model_t *m, uint32_t token, uint32_t pos, uint32_t n_src)
{
    it2_scratch_t *s = scratch_of(m);
    float *x = s->x, *h = s->h, *q = s->q, *a = s->k;
    uint32_t l, i;

    q4_row(m, m->tgt_emb, token, sqrtf((float)D), x);
    add_position(pos + 2U, x);
    layer_norm(&m->dec_emb_norm, x, 1, x);
    for (l = 0; l < m->hdr->n_dec; ++l)
    {
        const it2_dec_layer_t *L = &m->dec[l];
        float *sk = s->self_k + (size_t)l * IT2_MAX_TGT * D;
        float *sv = s->self_v + (size_t)l * IT2_MAX_TGT * D;

        layer_norm(&L->sa_norm, x, 1, h);
        linear(m, &L->q, h, 1, q);
        linear(m, &L->k, h, 1, sk + (size_t)pos * D);
        linear(m, &L->v, h, 1, sv + (size_t)pos * D);
        attention(q, 1, sk, sv, pos + 1U, s->att, a);
        linear(m, &L->o, a, 1, q);
        for (i = 0; i < D; ++i)
        {
            x[i] += q[i];
        }
        layer_norm(&L->ca_norm, x, 1, h);
        linear(m, &L->cq, h, 1, q);
        attention(q, 1, s->cross_k + (size_t)l * IT2_MAX_SRC * D, s->cross_v + (size_t)l * IT2_MAX_SRC * D, n_src,
                  s->att, a);
        linear(m, &L->co, a, 1, q);
        for (i = 0; i < D; ++i)
        {
            x[i] += q[i];
        }
        layer_norm(&L->ff_norm, x, 1, h);
        linear(m, &L->fc1, h, 1, s->ffn);
        gelu_inplace(s, s->ffn, FFN);
        linear(m, &L->fc2, s->ffn, 1, q);
        for (i = 0; i < D; ++i)
        {
            x[i] += q[i];
        }
    }
    layer_norm(&m->dec_final_norm, x, 1, h);
    {
        it2_linear_t out = {m->tgt_emb, NULL};
        linear(m, &out, h, 1, s->logits);
    }
}

/* ------------------------------------------------------------------------------------------ */
/*                                   English tokenisation                                     */
/* ------------------------------------------------------------------------------------------ */

typedef struct
{
    uint32_t off, len;
    float score;
    uint32_t id;
} it2_src_piece_t;

typedef struct
{
    uint32_t off, len;
} it2_tgt_piece_t;

static int is_alpha(unsigned char c)
{
    return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z'));
}

static int is_alnum(unsigned char c)
{
    return is_alpha(c) || ((c >= '0') && (c <= '9'));
}

/**
 * @brief Moses-style English tokenisation for ASR text: punctuation becomes its own token, and an
 *        apostrophe between letters starts the next token ("don't" -> "don 't").
 */
static uint32_t moses_tokenize(const char *in, char *out, uint32_t cap)
{
    uint32_t n = 0;
    const unsigned char *p = (const unsigned char *)in;
    int pending_space = 0;

#define EMIT(ch)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((n + 2U) < cap)                                                                                            \
        {                                                                                                              \
            out[n++] = (char)(ch);                                                                                     \
        }                                                                                                              \
    } while (0)
#define SPACE()                                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((n > 0U) && (out[n - 1U] != ' '))                                                                          \
        {                                                                                                              \
            EMIT(' ');                                                                                                 \
        }                                                                                                              \
    } while (0)

    for (; *p != '\0'; ++p)
    {
        const unsigned char c = *p;

        if ((c == ' ') || (c == '\t') || (c == '\n') || (c == '\r'))
        {
            pending_space = 1;
            continue;
        }
        if (pending_space)
        {
            SPACE();
            pending_space = 0;
        }
        if (c == '\'')
        {
            /* letter'letter: split before the apostrophe; otherwise the apostrophe stands alone */
            const int prev_alpha = (p > (const unsigned char *)in) && is_alnum(p[-1]);
            const int next_alpha = is_alpha(p[1]);
            if (prev_alpha && next_alpha)
            {
                SPACE();
                EMIT('\'');
            }
            else
            {
                SPACE();
                EMIT('\'');
                EMIT(' ');
            }
            continue;
        }
        if ((c == ',') || (c == '?') || (c == '!') || (c == ';') || (c == ':') || (c == '"') || (c == '(') ||
            (c == ')'))
        {
            SPACE();
            EMIT(c);
            EMIT(' ');
            continue;
        }
        if (c == '.')
        {
            /* a period between digits or letters stays (3.5, e.g.); otherwise it is a token */
            const int prev_an = (p > (const unsigned char *)in) && is_alnum(p[-1]);
            const int next_an = is_alnum(p[1]);
            if (prev_an && next_an)
            {
                EMIT('.');
            }
            else
            {
                SPACE();
                EMIT('.');
                EMIT(' ');
            }
            continue;
        }
        EMIT(c);
    }
    while ((n > 0U) && (out[n - 1U] == ' '))
    {
        n--;
    }
    out[n] = '\0';
#undef EMIT
#undef SPACE
    return n;
}

/** @brief Binary search of a byte string among the sorted source pieces; -1 if absent. */
static int32_t find_piece(const it2_model_t *m, const uint8_t *str, uint32_t len)
{
    const it2_src_piece_t *pieces = (const it2_src_piece_t *)(const void *)(m->blob + m->hdr->src_off);
    const uint8_t *strings = m->blob + m->hdr->src_str_off;
    int32_t lo = 0, hi = (int32_t)m->hdr->n_src_pieces - 1;

    while (lo <= hi)
    {
        const int32_t mid = (lo + hi) / 2;
        const it2_src_piece_t *e = &pieces[mid];
        const uint32_t common = (e->len < len) ? e->len : len;
        int cmp = memcmp(strings + e->off, str, common);

        if (cmp == 0)
        {
            cmp = (e->len < len) ? -1 : ((e->len > len) ? 1 : 0);
        }
        if (cmp == 0)
        {
            return mid;
        }
        if (cmp < 0)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid - 1;
        }
    }
    return -1;
}

static uint32_t utf8_len(uint8_t c)
{
    return (c < 0x80U) ? 1U : ((c < 0xE0U) ? 2U : ((c < 0xF0U) ? 3U : 4U));
}

#define IT2_MAX_WORD 96U

/** @brief SentencePiece BPE on one word (already prefixed with U+2581): append its ids. */
static uint32_t bpe_word(const it2_model_t *m, const uint8_t *w, uint32_t wlen, uint32_t *ids, uint32_t max)
{
    const it2_src_piece_t *pieces = (const it2_src_piece_t *)(const void *)(m->blob + m->hdr->src_off);
    uint16_t start[IT2_MAX_WORD], len[IT2_MAX_WORD];
    uint32_t n = 0, i, out = 0;

    for (i = 0; (i < wlen) && (n < IT2_MAX_WORD); n++)
    {
        start[n] = (uint16_t)i;
        len[n] = (uint16_t)utf8_len(w[i]);
        i += len[n];
    }
    for (;;)
    {
        int32_t best = -1;
        float best_score = -1e30f;

        for (i = 0; (i + 1U) < n; ++i)
        {
            const int32_t p = find_piece(m, w + start[i], (uint32_t)len[i] + len[i + 1U]);
            if ((p >= 0) && (pieces[p].score > best_score))
            {
                best_score = pieces[p].score;
                best = (int32_t)i;
            }
        }
        if (best < 0)
        {
            break;
        }
        len[best] = (uint16_t)(len[best] + len[best + 1]);
        for (i = (uint32_t)best + 1U; (i + 1U) < n; ++i)
        {
            start[i] = start[i + 1U];
            len[i] = len[i + 1U];
        }
        n--;
    }
    for (i = 0; (i < n) && (out < max); ++i)
    {
        const int32_t p = find_piece(m, w + start[i], len[i]);
        ids[out++] = (p >= 0) ? pieces[p].id : m->hdr->unk;
    }
    return out;
}

uint32_t it2_encode_text(const it2_model_t *m, const char *english, uint32_t *ids, uint32_t max)
{
    char tok[512];
    uint8_t word[IT2_MAX_WORD * 4U];
    uint32_t n = 0;
    const char *p;

    if (max < 3U)
    {
        return 0;
    }
    ids[n++] = m->hdr->tag_src;
    ids[n++] = m->hdr->tag_tgt;
    (void)moses_tokenize(english, tok, sizeof(tok));
    p = tok;
    while (*p != '\0')
    {
        uint32_t wl = 3;

        while (*p == ' ')
        {
            p++;
        }
        if (*p == '\0')
        {
            break;
        }
        word[0] = 0xE2U; /* U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's word start */
        word[1] = 0x96U;
        word[2] = 0x81U;
        while ((*p != '\0') && (*p != ' ') && (wl < sizeof(word)))
        {
            word[wl++] = (uint8_t)*p++;
        }
        n += bpe_word(m, word, wl, ids + n, max - 1U - n);
    }
    ids[n++] = m->hdr->eos;
    return n;
}

/* ------------------------------------------------------------------------------------------ */
/*                                     Hindi detokenising                                     */
/* ------------------------------------------------------------------------------------------ */

static int is_punct_after(const char *s)
{
    /* , . ? ! ; : and the danda U+0964 (E0 A5 A4) take no space before them */
    return (*s == ',') || (*s == '.') || (*s == '?') || (*s == '!') || (*s == ';') || (*s == ':') ||
           (((uint8_t)s[0] == 0xE0U) && ((uint8_t)s[1] == 0xA5U) && ((uint8_t)s[2] == 0xA4U));
}

static int detokenize(const it2_model_t *m, const uint32_t *toks, uint32_t n, char *out, size_t cap)
{
    const it2_tgt_piece_t *pieces = (const it2_tgt_piece_t *)(const void *)(m->blob + m->hdr->tgt_off);
    const char *strings = (const char *)(m->blob + m->hdr->tgt_str_off);
    char tmp[1024];
    size_t len = 0, i, o = 0;
    uint32_t t;

    for (t = 0; t < n; ++t)
    {
        const it2_tgt_piece_t *e = &pieces[toks[t]];
        uint32_t j = 0;

        while ((j < e->len) && ((len + 1U) < sizeof(tmp)))
        {
            if (((uint8_t)strings[e->off + j] == 0xE2U) && ((j + 2U) < e->len) &&
                ((uint8_t)strings[e->off + j + 1U] == 0x96U) && ((uint8_t)strings[e->off + j + 2U] == 0x81U))
            {
                tmp[len++] = ' ';
                j += 3U;
            }
            else
            {
                tmp[len++] = strings[e->off + j++];
            }
        }
    }
    tmp[len] = '\0';
    /* trim, collapse spaces, no space before punctuation, "word - word" -> "word-word" */
    for (i = 0; (i < len) && (tmp[i] == ' '); ++i)
    {
    }
    for (; (i < len) && ((o + 1U) < cap); ++i)
    {
        if (tmp[i] == ' ')
        {
            if ((o == 0U) || (out[o - 1U] == ' ') || (tmp[i + 1U] == '\0') || (tmp[i + 1U] == ' ') ||
                is_punct_after(&tmp[i + 1U]))
            {
                continue;
            }
            if ((tmp[i + 1U] == '-') && (tmp[i + 2U] == ' ') && (o > 0U) && (out[o - 1U] != ' '))
            {
                out[o++] = '-';
                i += 2U;
                continue;
            }
        }
        out[o++] = tmp[i];
    }
    while ((o > 0U) && (out[o - 1U] == ' '))
    {
        o--;
    }
    out[o] = '\0';
    return (int)o;
}

/* ------------------------------------------------------------------------------------------ */
/*                                       Public API                                           */
/* ------------------------------------------------------------------------------------------ */

int it2_open(it2_model_t *m, const void *blob, void *workspace)
{
    const it2_header_t *h = (const it2_header_t *)blob;
    const it2_tensor_t *t;
    uint8_t *ws = (uint8_t *)workspace;
    it2_scratch_t *s;
    uint32_t i, k = 0;

    if ((m == NULL) || (blob == NULL) || (workspace == NULL) || (memcmp(h->magic, "IT2B", 4) != 0) ||
        (h->version != 1U) || (h->d_model != D) || (h->n_heads != H) || (h->d_ffn != FFN) || (h->group != GROUP))
    {
        return -1;
    }
    memset(m, 0, sizeof(*m));
    m->blob = (const uint8_t *)blob;
    m->hdr = h;
    t = (const it2_tensor_t *)(const void *)(m->blob + 256);
    if (h->n_tensors != (5U + (h->n_enc * 16U) + 5U + (h->n_dec * 26U)))
    {
        return -2;
    }

#define F32(idx) ((const float *)(const void *)(m->blob + t[(idx)].data_off))
#define NORM(dst)                                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        (dst).w = F32(k);                                                                                              \
        (dst).b = F32(k + 1U);                                                                                         \
        k += 2U;                                                                                                       \
    } while (0)
#define LIN(dst)                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        (dst).w = &t[k];                                                                                               \
        (dst).b = F32(k + 1U);                                                                                         \
        k += 2U;                                                                                                       \
    } while (0)

    m->enc = (it2_enc_layer_t *)(void *)ws;
    ws += align32(h->n_enc * sizeof(it2_enc_layer_t));
    m->dec = (it2_dec_layer_t *)(void *)ws;
    ws += align32(h->n_dec * sizeof(it2_dec_layer_t));

    m->src_emb = &t[k++];
    NORM(m->enc_emb_norm);
    NORM(m->enc_final_norm);
    for (i = 0; i < h->n_enc; ++i)
    {
        it2_enc_layer_t *L = &m->enc[i];
        NORM(L->sa_norm);
        LIN(L->q);
        LIN(L->k);
        LIN(L->v);
        LIN(L->o);
        NORM(L->ff_norm);
        LIN(L->fc1);
        LIN(L->fc2);
    }
    m->tgt_emb = &t[k++];
    NORM(m->dec_emb_norm);
    NORM(m->dec_final_norm);
    for (i = 0; i < h->n_dec; ++i)
    {
        it2_dec_layer_t *L = &m->dec[i];
        NORM(L->sa_norm);
        LIN(L->q);
        LIN(L->k);
        LIN(L->v);
        LIN(L->o);
        NORM(L->ca_norm);
        LIN(L->cq);
        LIN(L->ck);
        LIN(L->cv);
        LIN(L->co);
        NORM(L->ff_norm);
        LIN(L->fc1);
        LIN(L->fc2);
    }
#undef F32
#undef NORM
#undef LIN

    s = (it2_scratch_t *)(void *)ws;
    ws += align32(sizeof(it2_scratch_t));
#define TAKE(ptr, type, bytes)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        (ptr) = (type *)(void *)ws;                                                                                    \
        ws += align32(bytes);                                                                                          \
    } while (0)
    TAKE(s->x, float, IT2_MAX_SRC * D * sizeof(float));
    TAKE(s->h, float, IT2_MAX_SRC * D * sizeof(float));
    TAKE(s->q, float, IT2_MAX_SRC * D * sizeof(float));
    TAKE(s->k, float, IT2_MAX_SRC * D * sizeof(float));
    TAKE(s->v, float, IT2_MAX_SRC * D * sizeof(float));
    TAKE(s->att, float, H * IT2_MAX_TGT * sizeof(float));
    TAKE(s->ffn, float, IT2_MAX_SRC * FFN * sizeof(float));
    TAKE(s->xq, int8_t, IT2_MAX_SRC * FFN);
    TAKE(s->xs, float, IT2_MAX_SRC * (FFN / GROUP) * sizeof(float));
    TAKE(s->cross_k, float, (size_t)h->n_dec * IT2_MAX_SRC * D * sizeof(float));
    TAKE(s->cross_v, float, (size_t)h->n_dec * IT2_MAX_SRC * D * sizeof(float));
    TAKE(s->self_k, float, (size_t)h->n_dec * IT2_MAX_TGT * D * sizeof(float));
    TAKE(s->self_v, float, (size_t)h->n_dec * IT2_MAX_TGT * D * sizeof(float));
    TAKE(s->logits, float, h->tgt_vocab * sizeof(float));
    TAKE(s->gelu, float, (GELU_STEPS + 1U) * sizeof(float));
    TAKE(s->src_ids, uint32_t, IT2_MAX_SRC * sizeof(uint32_t));
#undef TAKE
    for (i = 0; i <= GELU_STEPS; ++i)
    {
        const float x = -GELU_RANGE + ((2.0f * GELU_RANGE * (float)i) / (float)GELU_STEPS);
        s->gelu[i] = 0.5f * x * (1.0f + erff(x * 0.70710678f));
    }
    return 0;
}

int it2_translate(const it2_model_t *m, const char *english, char *hindi, size_t capacity, it2_stats_t *stats)
{
    it2_scratch_t *s;
    uint32_t out[IT2_MAX_TGT];
    uint32_t n_src, n_out = 0, pos, tok, repeat = 0, max_len;
    uint32_t t0, t1;

    if ((m == NULL) || (english == NULL) || (hindi == NULL) || (capacity == 0U))
    {
        return -1;
    }
    s = scratch_of(m);
    hindi[0] = '\0';
    n_src = it2_encode_text(m, english, s->src_ids, IT2_MAX_SRC);
    if (n_src <= 3U)
    {
        return 0; /* nothing but the tags and </s> */
    }
    t0 = it2_now_ms();
    encode(m, s->src_ids, n_src);
    t1 = it2_now_ms();
    max_len = (2U * n_src) + 16U;
    if (max_len > IT2_MAX_TGT)
    {
        max_len = IT2_MAX_TGT;
    }
    tok = m->hdr->eos; /* decoder start token */
    for (pos = 0; pos < max_len; ++pos)
    {
        uint32_t v, best = m->hdr->eos;
        float best_v = -1e30f;

        decode_step(m, tok, pos, n_src);
        for (v = 0; v < m->hdr->tgt_vocab; ++v)
        {
            if ((v == m->hdr->bos) || (v == m->hdr->pad) || (v == m->hdr->unk))
            {
                continue;
            }
            if (s->logits[v] > best_v)
            {
                best_v = s->logits[v];
                best = v;
            }
        }
        if (best == m->hdr->eos)
        {
            break;
        }
        repeat = (best == tok) ? (repeat + 1U) : 0U;
        if (repeat >= IT2_MAX_REPEAT)
        {
            break;
        }
        out[n_out++] = best;
        tok = best;
    }
    if (stats != NULL)
    {
        stats->src_tokens = n_src;
        stats->tgt_tokens = n_out;
        stats->encode_ms = t1 - t0;
        stats->decode_ms = it2_now_ms() - t1;
    }
    return detokenize(m, out, n_out, hindi, capacity);
}
