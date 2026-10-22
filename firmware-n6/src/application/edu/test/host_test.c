/**
 * @file     host_test.c
 * @brief    Host test of edu_math.c: every vector from gen_vectors.py must match mathspeech.py.
 * @details  Usage: host_test VECTORS.tsv [max_failures_shown]
 *           Prints agreement per category, the first mismatches, and the slowest call.
 *           Exit code 0 only when every vector agrees.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "edu_math.h"

#define MAX_LINE   8192
#define MAX_FIELDS 32
#define MAX_CATS   64
#define MAX_CLAIMS 64

typedef struct
{
    char name[32];
    long total;
    long failed;
    double seconds;
} category_t;

static edu_math_work_t work;
static char line[MAX_LINE];
static char got[MAX_LINE];
static category_t cats[MAX_CATS];
static int cat_count;
#define SLOW_KEPT 10
static double slow_time[SLOW_KEPT];
static char slow_line[SLOW_KEPT][MAX_LINE];
static char raw[MAX_LINE];

static category_t *category(const char *name)
{
    int i;

    for (i = 0; i < cat_count; ++i)
    {
        if (strcmp(cats[i].name, name) == 0)
        {
            return &cats[i];
        }
    }
    if (cat_count == MAX_CATS)
    {
        return &cats[MAX_CATS - 1];
    }
    snprintf(cats[cat_count].name, sizeof(cats[cat_count].name), "%s", name);
    return &cats[cat_count++];
}

/** @brief Split on tabs in place, keeping empty fields. */
static int split(char *s, char **fields)
{
    int n = 0;

    fields[n++] = s;
    for (; *s != '\0'; ++s)
    {
        if ((*s == '\t') && (n < MAX_FIELDS))
        {
            *s = '\0';
            fields[n++] = s + 1;
        }
    }
    return n;
}

static int16_t opt(const char *s)
{
    return (strcmp(s, "-") == 0) ? (int16_t)EDU_MATH_NONE : (int16_t)atoi(s);
}

static void put_opt(char *buffer, size_t size, int value)
{
    if (value < 0)
    {
        snprintf(buffer, size, "-");
    }
    else
    {
        snprintf(buffer, size, "%d", value);
    }
}

static int parse_ranges(const char *s, edu_math_ranges_t *r)
{
    int a, b, c, d;

    if (strcmp(s, "-") == 0)
    {
        return 0;
    }
    if (sscanf(s, "%d,%d,%d,%d", &a, &b, &c, &d) != 4)
    {
        return 0;
    }
    r->table_min = (int16_t)a;
    r->table_max = (int16_t)b;
    r->mult_min = (int16_t)c;
    r->mult_max = (int16_t)d;
    return 1;
}

static double now(void)
{
    struct timespec ts;

    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/** @brief Run one vector; write the C output in the vector's format into got. Returns 1 on match. */
static int check(char **f, int n, const char **want_out, const char **text_out)
{
    edu_math_ranges_t r;
    int have_ranges;

    got[0] = '\0';
    if ((strcmp(f[0], "P") == 0) && (n >= 18))
    {
        edu_math_result_t res;
        int16_t expect[2];
        const int16_t *e = NULL;
        char a[8], b[8], c[8], k[8];
        const edu_math_claim_t *cl = &res.claim;

        have_ranges = parse_ranges(f[2], &r);
        if (strcmp(f[3], "-") != 0)
        {
            expect[0] = opt(f[3]);
            expect[1] = opt(f[4]);
            e = expect;
        }
        if (edu_math_parse(&work, f[5], e, have_ranges ? &r : NULL, &res) != EDU_MATH_OK)
        {
            snprintf(got, sizeof(got), "status error");
        }
        else
        {
            put_opt(a, sizeof(a), cl->a);
            put_opt(b, sizeof(b), cl->b);
            put_opt(c, sizeof(c), cl->c);
            put_opt(k, sizeof(k), cl->correct);
            snprintf(got, sizeof(got), "%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s", cl->template_id, a, b, c, k,
                     cl->ambiguous, res.doubt, cl->score, cl->span_first, cl->span_end, res.shown);
        }
        *text_out = f[5];
        /* the expected half starts at field 7; rejoin it with tabs */
        {
            static char want[MAX_LINE];
            int i;

            want[0] = '\0';
            for (i = 7; i < n; ++i)
            {
                strcat(want, f[i]);
                if (i + 1 < n)
                {
                    strcat(want, "\t");
                }
            }
            *want_out = want;
        }
        return strcmp(got, *want_out) == 0;
    }
    if ((strcmp(f[0], "M") == 0) && (n >= 9))
    {
        static edu_math_claim_t claims[MAX_CLAIMS];
        static char want[MAX_LINE];
        uint32_t count = 0U;
        uint32_t items = 0U;
        uint32_t i;
        size_t used;

        have_ranges = parse_ranges(f[2], &r);
        if (edu_math_parse_many(&work, f[4], opt(f[3]), have_ranges ? &r : NULL, claims, MAX_CLAIMS, &count, &items) !=
            EDU_MATH_OK)
        {
            snprintf(got, sizeof(got), "status error");
        }
        else
        {
            used = (size_t)snprintf(got, sizeof(got), "%u\t%u\t", (unsigned)items, (unsigned)count);
            for (i = 0U; (i < count) && (i < MAX_CLAIMS); ++i)
            {
                char a[8], b[8], c[8], k[8];

                put_opt(a, sizeof(a), claims[i].a);
                put_opt(b, sizeof(b), claims[i].b);
                put_opt(c, sizeof(c), claims[i].c);
                put_opt(k, sizeof(k), claims[i].correct);
                used += (size_t)snprintf(got + used, sizeof(got) - used, "%s%s,%s,%s,%s,%d,%d,%d,%d", (i > 0U) ? ";" : "",
                                         a, b, c, k, claims[i].ambiguous, claims[i].score, claims[i].span_first,
                                         claims[i].span_end);
            }
        }
        snprintf(want, sizeof(want), "%s\t%s\t%s", f[6], f[7], f[8]);
        *want_out = want;
        *text_out = f[4];
        return strcmp(got, want) == 0;
    }
    if ((strcmp(f[0], "C") == 0) && (n >= 5))
    {
        snprintf(got, sizeof(got), "%d", edu_math_may_continue(&work, f[2]));
        *want_out = f[4];
        *text_out = f[2];
        return strcmp(got, f[4]) == 0;
    }
    *want_out = "(malformed vector)";
    *text_out = "";
    return 0;
}

int main(int argc, char **argv)
{
    FILE *in;
    long total = 0;
    long failed = 0;
    int show = (argc > 2) ? atoi(argv[2]) : 25;
    int i;

    if ((argc < 2) || ((in = fopen(argv[1], "rb")) == NULL))
    {
        fprintf(stderr, "usage: host_test VECTORS.tsv [max_failures_shown]\n");
        return 2;
    }
    while (fgets(line, sizeof(line), in) != NULL)
    {
        char *f[MAX_FIELDS];
        size_t len = strlen(line);
        const char *want;
        const char *text;
        category_t *cat;
        double t0;
        double dt;
        int n;
        int ok;

        while ((len > 0U) && ((line[len - 1U] == '\n') || (line[len - 1U] == '\r')))
        {
            line[--len] = '\0';
        }
        if (len == 0U)
        {
            continue;
        }
        memcpy(raw, line, len + 1U);
        n = split(line, f);
        cat = category((n > 1) ? f[1] : "?");
        t0 = now();
        ok = check(f, n, &want, &text);
        dt = now() - t0;
        cat->seconds += dt;
        if (f[0][0] != 'C')
        {
            /* keep the slowest few; re-timed at the end */
            int s = 0;

            for (i = 1; i < SLOW_KEPT; ++i)
            {
                s = (slow_time[i] < slow_time[s]) ? i : s;
            }
            if (dt > slow_time[s])
            {
                slow_time[s] = dt;
                memcpy(slow_line[s], raw, len + 1U);
            }
        }
        ++total;
        ++cat->total;
        if (!ok)
        {
            ++failed;
            ++cat->failed;
            if (failed <= show)
            {
                printf("MISMATCH %s [%s] \"%s\"\n  python: %s\n  c:      %s\n", f[0], cat->name, text, want, got);
            }
        }
    }
    fclose(in);
    printf("\n%-18s %8s %8s %9s\n", "category", "vectors", "agree", "us/call");
    for (i = 0; i < cat_count; ++i)
    {
        printf("%-18s %8ld %7.3f%% %9.1f\n", cats[i].name, cats[i].total,
               100.0 * (double)(cats[i].total - cats[i].failed) / (double)cats[i].total,
               1e6 * cats[i].seconds / (double)cats[i].total);
    }
    printf("\nTOTAL %ld vectors, %ld agree, %ld differ: %.4f %% agreement with mathspeech.py\n", total, total - failed,
           failed, (total > 0) ? 100.0 * (double)(total - failed) / (double)total : 0.0);
    {
        /* the slowest calls again, best of 30, so a scheduler hiccup does not count */
        double worst = 0.0;
        int worst_at = 0;
        char *f[MAX_FIELDS];
        int n;

        for (i = 0; i < SLOW_KEPT; ++i)
        {
            double best = 1e9;
            int r;

            if (slow_time[i] <= 0.0)
            {
                continue;
            }
            for (r = 0; r < 30; ++r)
            {
                const char *want;
                const char *text;
                double t0;

                memcpy(line, slow_line[i], strlen(slow_line[i]) + 1U);
                n = split(line, f);
                t0 = now();
                (void)check(f, n, &want, &text);
                t0 = now() - t0;
                best = (t0 < best) ? t0 : best;
            }
            if (best > worst)
            {
                worst = best;
                worst_at = i;
            }
        }
        n = split(slow_line[worst_at], f);
        printf("slowest call on this PC (best of 30): %.3f ms (%s \"%s\")\n", worst * 1e3, f[0],
               (n > 5) ? ((f[0][0] == 'M') ? f[4] : f[5]) : "");
    }
    printf("work area: %u bytes\n", (unsigned)sizeof(edu_math_work_t));
    return (failed == 0 && total > 0) ? 0 : 1;
}
