/**
 * @file     edu_math.c
 * @brief    Spoken-multiplication checker; implements edu_math.h.
 * @details  Follows ASR_Math_Game/mathspeech.py function by function; the Python names are given
 *           in each comment. Three steps:
 *           1. Words: every word becomes a number word, an operator, an equals word or filler
 *              (exact, known misspelling, then edit distance; glued and split words repaired).
 *           2. Numbers: each run of number words is cut into numbers in every way spoken English
 *              allows ("ONE EIGHTY FOUR", "ONE HUNDRED AND EIGHTY FOUR", "ONE EIGHT FOUR").
 *           3. Check: the best-scoring reading, then A x B == C.
 *           Readings are enumerated depth first, in the order the Python lists them, so ties
 *           break the same way.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "edu_math.h"

#include <stddef.h>
#include <string.h>

#include "edu_math_tables.h"

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief Token kinds (mathspeech.Tok.kind). */
enum
{
    KIND_UNIT,
    KIND_TEEN,
    KIND_TENS,
    KIND_HUNDRED,
    KIND_A,
    KIND_OH,
    KIND_OP,
    KIND_EQ,
    KIND_BY,
    KIND_SKIP,
    KIND_FIX,
    KIND_UNKNOWN
};

/** @brief Number styles (mathspeech.STYLE_PENALTY keys). */
enum
{
    STYLE_STD,
    STYLE_COLLOQ,
    STYLE_DIGITS
};

/** @brief _parse_items_many: longest stretch of tokens one claim may cover. */
#define MANY_MAX_SPAN 18U

/** @brief At most this many numbers in one reading (_candidates max_numbers). */
#define MAX_NUMBERS 3U

/** @brief The longest number reading covers four words ("ONE HUNDRED TWENTY THREE"). */
#define MAX_NUMBER_WORDS 4U

/** @brief Template bit for edu_math_template_t t. */
#define TEMPLATE_BIT(t) ((uint8_t)(1U << (t)))

/** @brief All four templates. */
#define TEMPLATES_ALL (TEMPLATE_BIT(EDU_MATH_TEMPLATE_FULL) | TEMPLATE_BIT(EDU_MATH_TEMPLATE_ANSWER) | \
                       TEMPLATE_BIT(EDU_MATH_TEMPLATE_PARTIAL) | TEMPLATE_BIT(EDU_MATH_TEMPLATE_QUESTION))

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

static const edu_math_ranges_t default_ranges = { 2, 30, 1, 15 };

static const int16_t style_penalty[3] = { 0, 2, 5 };

static const char *const unit_words[10] = {
    "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"
};
static const char *const teen_words[10] = {
    "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN",
    "EIGHTEEN", "NINETEEN"
};
static const char *const tens_words[10] = {
    "", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"
};

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/* ------------------------------------------------------------------ words */

/** @brief strcmp(table word, s[0..n)). */
static int word_compare(const char *table_word, const char *s, uint32_t n)
{
    uint32_t i;

    for (i = 0U; i < n; ++i)
    {
        const int a = (unsigned char)table_word[i];
        const int b = (unsigned char)s[i];

        if (a != b)
        {
            return a - b; /* a == 0: the table word is shorter */
        }
    }
    return (table_word[n] == '\0') ? 0 : 1;
}

/** @brief Binary search of s[0..n) in a sorted word table; index or -1. */
static int find_word(const char (*table)[EDU_WORD_WIDTH], uint32_t count, const char *s, uint32_t n)
{
    uint32_t low = 0U;
    uint32_t high = count;

    if ((n == 0U) || (n >= EDU_WORD_WIDTH))
    {
        return -1;
    }
    while (low < high)
    {
        const uint32_t mid = (low + high) / 2U;
        const int order = word_compare(table[mid], s, n);

        if (order == 0)
        {
            return (int)mid;
        }
        if (order < 0)
        {
            low = mid + 1U;
        }
        else
        {
            high = mid;
        }
    }
    return -1;
}

static int known_find(const char *s, uint32_t n)
{
    return find_word(edu_known_words, EDU_KNOWN_COUNT, s, n);
}

static uint8_t word_class(int id)
{
    return edu_known[id].cls;
}

/** @brief s[0..n) equals the NUL-terminated word. */
static int word_is(const char *s, uint32_t n, const char *word)
{
    return (strlen(word) == n) && (memcmp(s, word, n) == 0);
}

/** @brief Exact or known misspelling -> known word, else -1 (Lexicon.canonical). */
static int canonical(const char *s, uint32_t n)
{
    int id = known_find(s, n);

    if (id < 0)
    {
        const int alias = find_word(edu_alias_words, EDU_ALIAS_COUNT, s, n);

        if (alias >= 0)
        {
            id = edu_alias_target[alias];
        }
    }
    return id;
}

/** @brief Levenshtein distance; both strings at most EDU_MAX_MATCH_LEN long (_edit_distance). */
static uint32_t edit_distance(const char *a, uint32_t la, const char *b, uint32_t lb)
{
    uint8_t prev[EDU_MAX_MATCH_LEN + 1U];
    uint8_t cur[EDU_MAX_MATCH_LEN + 1U];
    uint32_t i;
    uint32_t j;

    for (j = 0U; j <= lb; ++j)
    {
        prev[j] = (uint8_t)j;
    }
    for (i = 1U; i <= la; ++i)
    {
        cur[0] = (uint8_t)i;
        for (j = 1U; j <= lb; ++j)
        {
            uint32_t best = (uint32_t)prev[j] + 1U;
            const uint32_t insert = (uint32_t)cur[j - 1U] + 1U;
            const uint32_t change = (uint32_t)prev[j - 1U] + ((a[i - 1U] != b[j - 1U]) ? 1U : 0U);

            if (insert < best)
            {
                best = insert;
            }
            if (change < best)
            {
                best = change;
            }
            cur[j] = (uint8_t)best;
        }
        (void)memcpy(prev, cur, lb + 1U);
    }
    return prev[lb];
}

/** @brief Number of different meanings among the tied targets. */
static int meanings_differ(const uint8_t *ties, uint32_t count)
{
    uint32_t i;

    for (i = 1U; i < count; ++i)
    {
        if (edu_known[ties[i]].meaning != edu_known[ties[0]].meaning)
        {
            return 1;
        }
    }
    return 0;
}

/** @brief Canonical word, or the one fuzzy match all close targets agree on (Lexicon.lookup). */
static int lookup(const char *s, uint32_t n, uint8_t *fuzzy)
{
    uint8_t ties[EDU_TARGET_COUNT];
    uint32_t tie_count = 0U;
    uint32_t best = 0xFFU;
    uint32_t limit;
    uint32_t t;
    int hit = canonical(s, n);

    *fuzzy = 0U;
    if ((hit >= 0) || (n < 4U) || (n > EDU_MAX_MATCH_LEN))
    {
        return hit;
    }
    limit = (n <= 5U) ? 1U : 2U;
    for (t = 0U; t < EDU_TARGET_COUNT; ++t)
    {
        const char *target = edu_known_words[edu_targets[t]];
        const uint32_t length = (uint32_t)strlen(target);
        uint32_t d;

        if (((length > n) ? (length - n) : (n - length)) > limit)
        {
            continue;
        }
        d = edit_distance(s, n, target, length);
        if (d > limit)
        {
            continue;
        }
        if (d < best)
        {
            best = d;
            tie_count = 0U;
        }
        if (d == best)
        {
            ties[tie_count++] = edu_targets[t];
        }
    }
    if (tie_count == 0U)
    {
        return -1;
    }
    if (meanings_differ(ties, tie_count))
    {
        /* the recognizer drops letters more often than it adds them: "SEVENY" is SEVENTY */
        uint32_t kept = 0U;

        for (t = 0U; t < tie_count; ++t)
        {
            if (strlen(edu_known_words[ties[t]]) > n)
            {
                ties[kept++] = ties[t];
            }
        }
        tie_count = kept;
    }
    if ((tie_count == 0U) || meanings_differ(ties, tie_count))
    {
        return -1; /* two words just as close */
    }
    /* TWELVE or TWELVES for "TWELVS": keep the plural the recognizer wrote */
    for (t = 0U; t < tie_count; ++t)
    {
        const char *target = edu_known_words[ties[t]];

        if (target[strlen(target) - 1U] == s[n - 1U])
        {
            *fuzzy = 1U;
            return ties[t];
        }
    }
    *fuzzy = 1U;
    return ties[0];
}

/** @brief Class that never stands for a word of its own in a split or a join. */
static int is_marker_letter(int id)
{
    const uint8_t cls = word_class(id);

    return (cls == EDU_CLASS_A) || (cls == EDU_CLASS_OH) || (cls == EDU_CLASS_BY);
}

/**
 * @brief 'TWENTYTHREE' -> TWENTY THREE: the word is entirely known words of two letters or more
 *        (five or more may be misspelt) (Lexicon.split_glued).
 * @return Parts (two or more) appended to the entries, or 0 when it does not split.
 */
static uint32_t split_glued(edu_math_work_t *work, const char *s, uint32_t n, int *overflow)
{
    edu_math_split_t *best = work->split;
    uint32_t i;
    uint32_t j;
    uint32_t parts = 0U;

    for (i = 0U; i <= n; ++i)
    {
        best[i].set = 0U;
    }
    best[0].set = 1U;
    best[0].cost = 0;
    for (i = 0U; i < n; ++i)
    {
        const uint32_t last = ((i + EDU_MAX_MATCH_LEN) < n) ? (i + EDU_MAX_MATCH_LEN) : n;

        if (!best[i].set)
        {
            continue;
        }
        for (j = i + 2U; j <= last; ++j)
        {
            const char *part = &s[i];
            const uint32_t length = j - i;
            uint8_t fuzzy = 0U;
            int16_t cost;
            int hit;

            if (length == n)
            {
                continue; /* the whole word is lookup()'s job */
            }
            hit = known_find(part, length);
            if ((hit >= 0) && ((word_class(hit) == EDU_CLASS_FILLER) || (word_class(hit) == EDU_CLASS_FIX)))
            {
                if (!word_is(part, length, "AND"))
                {
                    continue;
                }
            }
            else
            {
                hit = canonical(part, length);
                if ((hit < 0) && (length >= 5U))
                {
                    hit = lookup(part, length, &fuzzy);
                }
                if ((hit < 0) || is_marker_letter(hit))
                {
                    continue;
                }
            }
            cost = (int16_t)(best[i].cost + 2 + fuzzy);
            if (!best[j].set || (cost < best[j].cost))
            {
                best[j].set = 1U;
                best[j].cost = cost;
                best[j].prev = (uint8_t)i;
                best[j].hit = (int16_t)hit;
                best[j].fuzzy = fuzzy;
            }
        }
    }
    if (!best[n].set)
    {
        return 0U;
    }
    for (j = n; j > 0U; j = best[j].prev)
    {
        ++parts;
    }
    if (parts < 2U)
    {
        return 0U;
    }
    if (work->entry_count + parts > EDU_MATH_MAX_TOKENS)
    {
        *overflow = 1;
        return 0U;
    }
    /* Walk back from the end, filling the parts in from the right. */
    i = work->entry_count + parts;
    for (j = n; j > 0U; j = best[j].prev)
    {
        edu_math_entry_t *entry = &work->entries[--i];

        entry->hit = best[j].hit;
        entry->fuzzy = best[j].fuzzy;
        entry->heard_to = (uint8_t)word_is(&s[best[j].prev], j - best[j].prev, "TO");
    }
    work->entry_count += parts;
    return parts;
}

/** @brief Append one entry; 0 when full. */
static int push_entry(edu_math_work_t *work, int hit, uint8_t fuzzy, uint8_t heard_to)
{
    edu_math_entry_t *entry;

    if (work->entry_count >= EDU_MATH_MAX_TOKENS)
    {
        return 0;
    }
    entry = &work->entries[work->entry_count++];
    entry->hit = (int16_t)hit;
    entry->fuzzy = fuzzy;
    entry->heard_to = heard_to;
    return 1;
}

/** @brief Append a spelt-out number to the normalized words (mathspeech.say, standard style). */
static int push_word(edu_math_work_t *work, const char *text, uint32_t length)
{
    if (work->word_count >= EDU_MATH_MAX_WORDS)
    {
        return 0;
    }
    work->words[work->word_count].text = text;
    work->words[work->word_count].length = (uint8_t)length;
    ++work->word_count;
    return 1;
}

static int push_const_word(edu_math_work_t *work, const char *word)
{
    return push_word(work, word, (uint32_t)strlen(word));
}

static int say_number(edu_math_work_t *work, uint32_t n)
{
    const uint32_t hundreds = n / 100U;
    const uint32_t rest = n % 100U;
    int ok = 1;

    if (n >= 1000U)
    {
        return 1; /* Python raises here; the board never writes digits */
    }
    if (n == 0U)
    {
        return push_const_word(work, unit_words[0]);
    }
    if (hundreds > 0U)
    {
        ok = push_const_word(work, unit_words[hundreds]) && push_const_word(work, "HUNDRED");
    }
    if (rest >= 20U)
    {
        ok = ok && push_const_word(work, tens_words[rest / 10U]);
        if ((rest % 10U) != 0U)
        {
            ok = ok && push_const_word(work, unit_words[rest % 10U]);
        }
    }
    else if (rest >= 10U)
    {
        ok = ok && push_const_word(work, teen_words[rest - 10U]);
    }
    else if (rest > 0U)
    {
        ok = ok && push_const_word(work, unit_words[rest]);
    }
    return ok;
}

/**
 * @brief Upper case, apostrophes dropped ("ONE'S" -> ONES), digits spelt out, x * = as TIMES and
 *        IS, anything else separates words (mathspeech.normalize).
 */
static edu_math_status_t normalize(edu_math_work_t *work, const char *text)
{
    const unsigned char *p = (const unsigned char *)text;
    uint32_t used = 0U;
    uint32_t start = 0U; /* first letter of the word being read */
    int in_word = 0;

    work->word_count = 0U;
    for (;;)
    {
        const unsigned char c = *p;
        const char *spelt = NULL;

        if ((c >= 'a') && (c <= 'z'))
        {
            if (used >= EDU_MATH_TEXT_CAPACITY)
            {
                return EDU_MATH_ERR_LIMIT;
            }
            work->text[used++] = (char)(c - 'a' + 'A');
            in_word = 1;
            ++p;
            continue;
        }
        if ((c >= 'A') && (c <= 'Z'))
        {
            if (used >= EDU_MATH_TEXT_CAPACITY)
            {
                return EDU_MATH_ERR_LIMIT;
            }
            work->text[used++] = (char)c;
            in_word = 1;
            ++p;
            continue;
        }
        if (c == '\'')
        {
            ++p; /* dropped: the letters around it join */
            continue;
        }
        if ((c == 0xE2U) && (p[1] == 0x80U) && (p[2] == 0x99U))
        {
            p += 3; /* right single quote, dropped like the apostrophe */
            continue;
        }

        /* Anything else ends the word being read. */
        if (in_word)
        {
            const uint32_t length = used - start;

            if (length > EDU_MATH_MAX_WORD_LEN)
            {
                return EDU_MATH_ERR_LIMIT;
            }
            if (!push_word(work, &work->text[start], length))
            {
                return EDU_MATH_ERR_LIMIT;
            }
            in_word = 0;
        }
        start = used;
        if (c == '\0')
        {
            break;
        }
        if ((c >= '0') && (c <= '9'))
        {
            uint32_t value = 0U;

            while ((*p >= '0') && (*p <= '9'))
            {
                value = value * 10U + (uint32_t)(*p - '0');
                if (value >= 1000U)
                {
                    value = 1000U; /* saturate: too large to say */
                }
                ++p;
            }
            if (!say_number(work, value))
            {
                return EDU_MATH_ERR_LIMIT;
            }
            continue;
        }
        if ((c == 0xC3U) && (p[1] == 0x97U))
        {
            spelt = "TIMES"; /* the multiplication sign */
            p += 2;
        }
        else if (c == '*')
        {
            spelt = "TIMES";
            ++p;
        }
        else if (c == '=')
        {
            spelt = "IS";
            ++p;
        }
        else
        {
            ++p;
        }
        if ((spelt != NULL) && !push_const_word(work, spelt))
        {
            return EDU_MATH_ERR_LIMIT;
        }
    }
    return EDU_MATH_OK;
}

/** @brief Heard words -> tokens; unknown words stay as KIND_UNKNOWN (mathspeech.tokenize). */
static edu_math_status_t tokenize(edu_math_work_t *work)
{
    const uint32_t count = work->word_count;
    uint32_t i = 0U;
    uint32_t k;
    int overflow = 0;

    work->entry_count = 0U;
    while (i < count)
    {
        const edu_math_word_t *w = &work->words[i];
        uint8_t fuzzy;
        const int hit = lookup(w->text, w->length, &fuzzy);

        if (i + 1U < count)
        {
            /* a word the recognizer split in two: "SEV EN", "FIF TEEN"; two good words ("SIX TEN")
               stay apart, and a join only matched loosely needs both halves unknown */
            const edu_math_word_t *next = &work->words[i + 1U];
            const uint32_t length = (uint32_t)w->length + next->length;
            uint8_t next_fuzzy;
            uint8_t join_fuzzy = 0U;
            const int next_hit = lookup(next->text, next->length, &next_fuzzy);
            int join_hit = -1;

            if (length <= EDU_MAX_MATCH_LEN)
            {
                char joined[EDU_MAX_MATCH_LEN];

                (void)memcpy(joined, w->text, w->length);
                (void)memcpy(&joined[w->length], next->text, next->length);
                join_hit = canonical(joined, length);
                if ((join_hit < 0) && (hit < 0) && (next_hit < 0))
                {
                    join_hit = lookup(joined, length, &join_fuzzy);
                }
            }
            if ((join_hit >= 0) && !is_marker_letter(join_hit) && (word_class(join_hit) != EDU_CLASS_FILLER) &&
                ((hit < 0) || (next_hit < 0) || fuzzy || next_fuzzy))
            {
                if (!push_entry(work, join_hit, join_fuzzy, 0U))
                {
                    return EDU_MATH_ERR_LIMIT;
                }
                i += 2U;
                continue;
            }
        }
        if ((hit < 0) || fuzzy)
        {
            /* several known words glued together beat one loosely matched word ("SIXTIMS") */
            const uint32_t before = work->entry_count;
            const uint32_t parts = split_glued(work, w->text, w->length, &overflow);

            if (overflow)
            {
                return EDU_MATH_ERR_LIMIT;
            }
            if (parts > 0U)
            {
                int any_fuzzy = 0;

                for (k = before; k < work->entry_count; ++k)
                {
                    any_fuzzy |= work->entries[k].fuzzy;
                }
                if ((hit < 0) || !any_fuzzy)
                {
                    ++i;
                    continue;
                }
                work->entry_count = before; /* keep the loose match instead */
            }
        }
        if (!push_entry(work, hit, fuzzy, (uint8_t)word_is(w->text, w->length, "TO")))
        {
            return EDU_MATH_ERR_LIMIT;
        }
        ++i;
    }

    for (k = 0U; k < work->entry_count; ++k)
    {
        const int id = work->entries[k].hit;
        const int next = (k + 1U < work->entry_count) ? work->entries[k + 1U].hit : -1;
        const int prev = (k > 0U) ? work->entries[k - 1U].hit : -1;
        edu_math_token_t *t = &work->tokens[k];

        t->plural = 0U;
        t->value = 0U;
        t->heard_to = work->entries[k].heard_to;
        if (id < 0)
        {
            t->kind = KIND_UNKNOWN;
            continue;
        }
        switch (word_class(id))
        {
        case EDU_CLASS_UNIT:
            t->kind = KIND_UNIT;
            break;
        case EDU_CLASS_TEEN:
            t->kind = KIND_TEEN;
            break;
        case EDU_CLASS_TENS:
            t->kind = KIND_TENS;
            break;
        case EDU_CLASS_HUNDRED:
            t->kind = KIND_HUNDRED;
            break;
        case EDU_CLASS_A:
            t->kind = ((next >= 0) && (word_class(next) == EDU_CLASS_HUNDRED)) ? KIND_A : KIND_SKIP;
            break;
        case EDU_CLASS_OH:
        {
            /* zero only inside "THREE OH FIVE" */
            const int ok = (prev >= 0) && (word_class(prev) == EDU_CLASS_UNIT) && !edu_known[prev].plural &&
                           (edu_known[prev].value > 0U) && (next >= 0) && (word_class(next) == EDU_CLASS_UNIT) &&
                           !edu_known[next].plural && (edu_known[next].value > 0U);

            t->kind = ok ? KIND_OH : KIND_SKIP;
            break;
        }
        case EDU_CLASS_BY:
            t->kind = KIND_BY;
            break;
        case EDU_CLASS_OP:
            t->kind = KIND_OP;
            break;
        case EDU_CLASS_EQ:
            t->kind = KIND_EQ;
            break;
        case EDU_CLASS_FIX:
            t->kind = KIND_FIX;
            break;
        default:
            t->kind = KIND_SKIP;
            break;
        }
        if ((t->kind == KIND_UNIT) || (t->kind == KIND_TEEN) || (t->kind == KIND_TENS) || (t->kind == KIND_HUNDRED) ||
            (t->kind == KIND_A))
        {
            t->value = edu_known[id].value;
            t->plural = edu_known[id].plural;
        }
    }
    /* "EQUAL TO": the TO is not a two. "MULTIPLIED BY": one operator. */
    for (k = 0U; k < work->entry_count; ++k)
    {
        edu_math_token_t *t = &work->tokens[k];

        if ((t->kind == KIND_UNIT) && t->heard_to && (k > 0U) && (work->tokens[k - 1U].kind == KIND_EQ))
        {
            t->kind = KIND_SKIP;
        }
        if (t->kind == KIND_BY)
        {
            t->kind = ((k > 0U) && (work->tokens[k - 1U].kind == KIND_OP)) ? KIND_OP : KIND_SKIP;
        }
    }
    work->token_count = work->entry_count;
    return EDU_MATH_OK;
}

static int is_number_kind(uint8_t kind)
{
    return kind <= KIND_OH;
}

static int is_marker(const edu_math_token_t *t)
{
    return (t->kind == KIND_OP) || (t->kind == KIND_EQ);
}

/** @brief Keep number words and markers; collapse repeated markers (mathspeech.clean). */
static void clean(edu_math_work_t *work, uint32_t from)
{
    uint32_t k;

    work->item_count = 0U;
    for (k = from; k < work->token_count; ++k)
    {
        const edu_math_token_t *t = &work->tokens[k];

        if (is_number_kind(t->kind))
        {
            work->items[work->item_count++] = *t;
        }
        else if (is_marker(t))
        {
            if ((work->item_count > 0U) && (work->items[work->item_count - 1U].kind == t->kind))
            {
                continue;
            }
            work->items[work->item_count++] = *t;
        }
    }
}

/* ---------------------------------------------------------------- numbers */

/** @brief Standard reading of the words as one number, or -1 (mathspeech._standard). */
static int32_t standard(const edu_math_token_t *t, uint32_t n)
{
    uint32_t i = 0U;
    int32_t total = 0;
    int hundreds = 0;

    if ((n == 1U) && (t[0].kind == KIND_UNIT) && (t[0].value == 0U))
    {
        return 0;
    }
    if ((n >= 2U) && (t[1].kind == KIND_HUNDRED) &&
        ((t[0].kind == KIND_A) || ((t[0].kind == KIND_UNIT) && (t[0].value > 0U))))
    {
        total = ((t[0].kind == KIND_A) ? 1 : (int32_t)t[0].value) * 100;
        i = 2U;
        hundreds = 1;
    }
    else if (t[0].kind == KIND_HUNDRED)
    {
        total = 100;
        i = 1U;
        hundreds = 1;
    }
    if (i == n)
    {
        return hundreds ? total : -1;
    }
    if (t[i].kind == KIND_TENS)
    {
        total += t[i].value;
        ++i;
        if ((i < n) && (t[i].kind == KIND_UNIT) && (t[i].value > 0U))
        {
            total += t[i].value;
            ++i;
        }
    }
    else if ((t[i].kind == KIND_TEEN) || ((t[i].kind == KIND_UNIT) && (t[i].value > 0U)))
    {
        total += t[i].value;
        ++i;
    }
    else
    {
        return -1;
    }
    return (i == n) ? total : -1;
}

/** @brief Every reading of these words as ONE number, std / colloq / digits (mathspeech.read_number). */
static uint32_t read_number(const edu_math_token_t *t, uint32_t n, int16_t *values, uint8_t *styles)
{
    uint32_t count = 0U;
    uint32_t i;
    int32_t v;

    if (n == 0U)
    {
        return 0U;
    }
    for (i = 0U; i + 1U < n; ++i)
    {
        if (t[i].plural)
        {
            return 0U;
        }
    }
    v = standard(t, n);
    if (v >= 0)
    {
        values[count] = (int16_t)v;
        styles[count++] = STYLE_STD;
    }
    if ((n >= 2U) && (t[0].kind == KIND_UNIT) && (t[0].value > 0U))
    {
        if (t[1].kind == KIND_OH)
        {
            if ((n == 3U) && (t[2].kind == KIND_UNIT) && (t[2].value > 0U))
            {
                values[count] = (int16_t)(t[0].value * 100U + t[2].value); /* THREE OH FIVE */
                styles[count++] = STYLE_COLLOQ;
            }
        }
        else if ((t[1].kind == KIND_TEEN) || (t[1].kind == KIND_TENS))
        {
            const int32_t r = standard(&t[1], n - 1U);

            if ((r >= 10) && (r <= 99))
            {
                values[count] = (int16_t)(t[0].value * 100U + (uint32_t)r); /* ONE EIGHTY FOUR */
                styles[count++] = STYLE_COLLOQ;
            }
        }
    }
    if ((n >= 2U) && (n <= 3U) && (t[0].kind == KIND_UNIT) && (t[0].value > 0U))
    {
        int all_digits = 1;
        int32_t digits = 0;

        for (i = 0U; i < n; ++i)
        {
            all_digits = all_digits && ((t[i].kind == KIND_UNIT) || (t[i].kind == KIND_OH));
            digits = digits * 10 + t[i].value;
        }
        if (all_digits)
        {
            values[count] = (int16_t)digits; /* ONE EIGHT FOUR */
            styles[count++] = STYLE_DIGITS;
        }
    }
    return count;
}

/* ----------------------------------------------------------------- claims */

static int fits(const edu_math_ranges_t *r, int32_t a, int32_t b)
{
    return ((r->table_min <= a) && (a <= r->table_max) && (r->mult_min <= b) && (b <= r->mult_max)) ||
           ((r->table_min <= b) && (b <= r->table_max) && (r->mult_min <= a) && (a <= r->mult_max));
}

static int same_pair(int32_t a, int32_t b, const int16_t *expect)
{
    return ((a == expect[0]) && (b == expect[1])) || ((a == expect[1]) && (b == expect[0]));
}

/**
 * @brief Template of a reading from where its markers and plurals sit (mathspeech._template).
 * @param k Numbers; op_mask / eq_mask: bit p set when a marker follows p numbers.
 */
static uint8_t template_of(uint32_t k, uint32_t op_mask, uint32_t eq_mask, uint32_t plural_mask)
{
    if ((k == 3U) && ((op_mask & ~2U) == 0U) && ((eq_mask & ~4U) == 0U) && ((plural_mask & ~2U) == 0U))
    {
        return EDU_MATH_TEMPLATE_FULL;
    }
    if ((k == 1U) && (op_mask == 0U) && ((eq_mask & ~1U) == 0U) && (plural_mask == 0U))
    {
        return EDU_MATH_TEMPLATE_ANSWER;
    }
    if ((k == 2U) && (eq_mask != 0U) && ((eq_mask & ~2U) == 0U) && ((op_mask & ~1U) == 0U) &&
        ((plural_mask & ~1U) == 0U))
    {
        return EDU_MATH_TEMPLATE_PARTIAL;
    }
    if ((k == 2U) && ((op_mask & ~2U) == 0U) && ((eq_mask & ~4U) == 0U) && ((plural_mask & ~2U) == 0U))
    {
        return EDU_MATH_TEMPLATE_QUESTION;
    }
    return EDU_MATH_TEMPLATE_NONE;
}

/** @brief (score, truth bonus) of a reading; the truth bonus only breaks ties (mathspeech._score). */
static void score_reading(const edu_math_work_t *work, uint8_t tpl, const edu_math_num_t *nums, uint32_t k,
                          int16_t *score, int16_t *truth)
{
    const int16_t *expect = work->e_expect;
    int32_t s = 0;
    int32_t bonus = 0;
    uint32_t i;

    for (i = 0U; i < k; ++i)
    {
        s -= style_penalty[nums[i].style];
    }
    if (tpl == EDU_MATH_TEMPLATE_FULL)
    {
        const int32_t a = nums[0].value;
        const int32_t b = nums[1].value;
        const int32_t c = nums[2].value;

        s += fits(work->e_ranges, a, b) ? 4 : -8;
        if (expect != NULL)
        {
            /* with a question on screen, other operands are more likely a misreading of an answer */
            s += same_pair(a, b, expect) ? 10 : -10;
        }
        bonus = (a * b == c) ? 3 : 0;
    }
    else if (tpl == EDU_MATH_TEMPLATE_ANSWER)
    {
        if (expect == NULL)
        {
            s -= 5; /* nothing to check it against */
        }
        else
        {
            bonus = ((int32_t)expect[0] * expect[1] == nums[0].value) ? 3 : 0;
        }
    }
    else if (tpl == EDU_MATH_TEMPLATE_PARTIAL)
    {
        s -= 2;
        if (expect != NULL)
        {
            const int32_t x = nums[0].value;
            const int in_question = (x == expect[0]) || (x == expect[1]);

            s += in_question ? 6 : -4;
            if (in_question && (x != 0))
            {
                const int32_t other = ((int32_t)expect[0] * expect[1]) / x;

                bonus = (x * other == nums[1].value) ? 3 : 0;
            }
        }
    }
    else
    {
        s -= 8;
    }
    *score = (int16_t)s;
    *truth = (int16_t)bonus;
}

/** @brief Claim of the best reading (mathspeech._make_claim). */
static void make_claim(const int16_t *expect, uint8_t tpl, const edu_math_num_t *nums, uint32_t k,
                       edu_math_claim_t *claim)
{
    claim->template_id = tpl;
    claim->ambiguous = 0U;
    claim->a = EDU_MATH_NONE;
    claim->b = EDU_MATH_NONE;
    claim->c = EDU_MATH_NONE;
    claim->correct = EDU_MATH_UNKNOWN;
    claim->score = 0;
    claim->span_first = nums[0].first;
    claim->span_end = (uint8_t)(nums[k - 1U].last + 1U);
    if (tpl == EDU_MATH_TEMPLATE_FULL)
    {
        claim->a = nums[0].value;
        claim->b = nums[1].value;
        claim->c = nums[2].value;
        claim->correct = (int8_t)((int32_t)claim->a * claim->b == claim->c);
        if ((expect != NULL) && !same_pair(claim->a, claim->b, expect))
        {
            claim->correct = 0;
        }
    }
    else if (tpl == EDU_MATH_TEMPLATE_ANSWER)
    {
        claim->c = nums[0].value;
        if (expect != NULL)
        {
            claim->a = expect[0];
            claim->b = expect[1];
            claim->correct = (int8_t)((int32_t)expect[0] * expect[1] == claim->c);
        }
    }
    else if (tpl == EDU_MATH_TEMPLATE_PARTIAL)
    {
        const int16_t x = nums[0].value;

        claim->b = x;
        claim->c = nums[1].value;
        if ((expect != NULL) && ((x == expect[0]) || (x == expect[1])))
        {
            claim->a = (x == expect[0]) ? expect[1] : expect[0];
            claim->correct = (int8_t)((int32_t)claim->a * x == claim->c);
        }
        else if (expect != NULL)
        {
            claim->correct = 0;
        }
    }
    else
    {
        claim->a = nums[0].value;
        claim->b = nums[1].value;
    }
}

/** @brief One complete reading: pass 1 keeps the best, pass 2 looks for a rival as good. */
static void visit(edu_math_work_t *work, uint32_t k, uint32_t op_mask, uint32_t eq_mask)
{
    uint32_t plural_mask = 0U;
    uint32_t i;
    uint8_t tpl;
    int16_t s;
    int16_t truth;

    for (i = 0U; i < k; ++i)
    {
        plural_mask |= (uint32_t)work->e_nums[i].plural << i;
    }
    tpl = template_of(k, op_mask, eq_mask, plural_mask);
    if ((tpl == EDU_MATH_TEMPLATE_NONE) || ((work->e_templates & TEMPLATE_BIT(tpl)) == 0U))
    {
        return;
    }
    score_reading(work, tpl, work->e_nums, k, &s, &truth);
    if (work->e_pass == 1U)
    {
        const int16_t total = (int16_t)(s + truth);

        /* strictly better only: the first of equal readings wins, as Python's stable sort */
        if (!work->e_found || (total > work->e_best_total))
        {
            work->e_found = 1U;
            work->e_best_total = total;
            work->e_best_base = s;
            work->e_best_template = tpl;
            work->e_best_count = (uint8_t)k;
            (void)memcpy(work->e_best_nums, work->e_nums, sizeof(work->e_best_nums));
        }
        return;
    }
    if (s >= work->e_best_base)
    {
        int same = (tpl == work->e_best_template);

        for (i = 0U; same && (i < k); ++i)
        {
            same = (work->e_nums[i].value == work->e_best_nums[i].value);
        }
        if (!same)
        {
            work->e_ambiguous = 1U;
        }
    }
}

/**
 * @brief Walk every reading of the tokens from pos on, depth first, in Python's order
 *        (mathspeech._candidates / _split_run / _structure).
 * @details Markers are recorded by how many numbers precede them. A number ends at a marker, at
 *          a plural, or when the words stop forming a number; every word must be used. Recursion
 *          is one level per number: at most MAX_NUMBERS + 1 frames.
 */
static void enumerate(edu_math_work_t *work, uint32_t pos, uint32_t k, uint32_t op_mask, uint32_t eq_mask)
{
    const edu_math_token_t *items = work->e_items;
    const uint32_t n = work->e_count;
    uint32_t end;
    uint32_t cut;
    uint32_t max_cut;

    while ((pos < n) && is_marker(&items[pos]))
    {
        if (items[pos].kind == KIND_OP)
        {
            op_mask |= 1U << k;
        }
        else
        {
            eq_mask |= 1U << k;
        }
        ++pos;
    }
    if (pos == n)
    {
        visit(work, k, op_mask, eq_mask);
        return;
    }
    if (k == MAX_NUMBERS)
    {
        return; /* words left over and no room for another number */
    }
    /* the run of number words: up to a marker, or through a plural ("TWENTY THREE EIGHTS") */
    end = pos;
    while ((end < n) && !is_marker(&items[end]))
    {
        ++end;
        if (items[end - 1U].plural)
        {
            break;
        }
    }
    max_cut = end - pos;
    if (max_cut > MAX_NUMBER_WORDS)
    {
        max_cut = MAX_NUMBER_WORDS; /* longer cuts never read as one number */
    }
    for (cut = 1U; cut <= max_cut; ++cut)
    {
        int16_t values[3];
        uint8_t styles[3];
        const uint32_t readings = read_number(&items[pos], cut, values, styles);
        uint32_t r;

        for (r = 0U; r < readings; ++r)
        {
            edu_math_num_t *num = &work->e_nums[k];

            num->value = values[r];
            num->style = styles[r];
            num->plural = items[pos + cut - 1U].plural;
            num->first = (uint8_t)pos;
            num->last = (uint8_t)(pos + cut - 1U);
            enumerate(work, pos + cut, k + 1U, op_mask, eq_mask);
        }
    }
}

/**
 * @brief Best claim over cleaned tokens (mathspeech._best).
 * @param ambiguity 1 to also look for a rival reading as good (Claim.ambiguous).
 * @return 1 with *claim filled, or 0 when no reading has an allowed template.
 */
static int best_claim(edu_math_work_t *work, const edu_math_token_t *items, uint32_t count, const int16_t *expect,
                      const edu_math_ranges_t *ranges, uint8_t templates, int ambiguity, edu_math_claim_t *claim)
{
    work->e_items = items;
    work->e_count = count;
    work->e_expect = expect;
    work->e_ranges = ranges;
    work->e_templates = templates;
    work->e_found = 0U;
    work->e_ambiguous = 0U;
    work->e_pass = 1U;
    enumerate(work, 0U, 0U, 0U, 0U);
    if (!work->e_found)
    {
        return 0;
    }
    if (ambiguity)
    {
        work->e_pass = 2U;
        enumerate(work, 0U, 0U, 0U, 0U);
    }
    make_claim(expect, work->e_best_template, work->e_best_nums, work->e_best_count, claim);
    claim->score = work->e_best_total;
    claim->ambiguous = work->e_ambiguous;
    return 1;
}

static void none_claim(edu_math_claim_t *claim)
{
    (void)memset(claim, 0, sizeof(*claim));
    claim->template_id = EDU_MATH_TEMPLATE_NONE;
    claim->correct = EDU_MATH_UNKNOWN;
    claim->a = EDU_MATH_NONE;
    claim->b = EDU_MATH_NONE;
    claim->c = EDU_MATH_NONE;
}

/**
 * @brief Best sequence of full claims over work->items (mathspeech._parse_items_many), search only.
 * @details Dynamic programming over cut points. A word left out costs more than anything a claim
 *          earns by being true, so a wrong line is never made right by dropping a word; each TIMES
 *          or IS used inside a claim earns a little, so cuts follow the speaker's own markers.
 *          Walk the result back from j = item_count with dp_prev; dp_has[j] marks a claim ending
 *          at j (fetch it with many_claim()).
 */
static void many_search(edu_math_work_t *work, const int16_t *expect, const edu_math_ranges_t *ranges,
                        int16_t table)
{
    const edu_math_token_t *items = work->items;
    const uint32_t n = work->item_count;
    uint32_t i;
    uint32_t j;

    work->dp_score[0] = 0;
    work->dp_has[0] = 0U;
    work->dp_prev[0] = 0U;
    for (j = 1U; j <= n; ++j)
    {
        int32_t best = work->dp_score[j - 1U] - 8; /* skip one token */
        uint32_t best_from = j - 1U;
        uint8_t has = 0U;
        const uint32_t low = (j > MANY_MAX_SPAN) ? (j - MANY_MAX_SPAN) : 0U;

        for (i = low; i + 2U < j; ++i)
        {
            edu_math_claim_t c;
            uint32_t markers = 0U;
            uint32_t m;
            int32_t total;

            if (!best_claim(work, &items[i], j - i, expect, ranges, TEMPLATE_BIT(EDU_MATH_TEMPLATE_FULL), 0, &c))
            {
                continue;
            }
            for (m = c.span_first + i; m < c.span_end + i; ++m)
            {
                markers += is_marker(&items[m]) ? 1U : 0U;
            }
            /* a small reward per claim: larger ones favour many short, wrong claims */
            total = work->dp_score[i] + c.score + 2 + 2 * (int32_t)markers +
                    (((table != EDU_MATH_NONE) && ((c.a == table) || (c.b == table))) ? 6 : 0);
            if (total > best)
            {
                best = total;
                best_from = i;
                has = 1U;
            }
        }
        work->dp_score[j] = best;
        work->dp_prev[j] = (uint8_t)best_from;
        work->dp_has[j] = has;
    }
}

/** @brief The claim many_search() found ending at j, with the rival check, span in work->items. */
static void many_claim(edu_math_work_t *work, uint32_t j, const int16_t *expect, const edu_math_ranges_t *ranges,
                       edu_math_claim_t *claim)
{
    const uint32_t from = work->dp_prev[j];

    (void)best_claim(work, &work->items[from], j - from, expect, ranges, TEMPLATE_BIT(EDU_MATH_TEMPLATE_FULL), 1,
                     claim);
    claim->span_first = (uint8_t)(claim->span_first + from);
    claim->span_end = (uint8_t)(claim->span_end + from);
}

/** @brief One utterance, words already normalized (mathspeech._parse_once). */
static edu_math_status_t parse_once(edu_math_work_t *work, const int16_t *expect, const edu_math_ranges_t *ranges,
                                    edu_math_claim_t *claim)
{
    uint32_t starts[2];
    uint32_t attempts = 0U;
    uint32_t a;
    uint32_t k;
    int last_fix = -1;
    edu_math_status_t status;

    none_claim(claim);
    status = tokenize(work);
    if (status != EDU_MATH_OK)
    {
        return status;
    }
    for (k = 0U; k < work->token_count; ++k)
    {
        if (work->tokens[k].kind == KIND_FIX)
        {
            last_fix = (int)k;
        }
    }
    if (last_fix >= 0)
    {
        starts[attempts++] = (uint32_t)last_fix + 1U; /* what came after the correction */
    }
    starts[attempts++] = 0U;
    for (a = 0U; a < attempts; ++a)
    {
        clean(work, starts[a]);
        if (work->item_count == 0U)
        {
            continue;
        }
        if (best_claim(work, work->items, work->item_count, expect, ranges, TEMPLATES_ALL, 1, claim))
        {
            return EDU_MATH_OK;
        }
        if (work->item_count > 3U)
        {
            /* several sums in one breath: take the last complete one */
            uint32_t j;

            many_search(work, expect, ranges, EDU_MATH_NONE);
            for (j = work->item_count; j > 0U; j = work->dp_prev[j])
            {
                if (work->dp_has[j])
                {
                    many_claim(work, j, expect, ranges, claim);
                    return EDU_MATH_OK;
                }
            }
        }
    }
    none_claim(claim);
    return EDU_MATH_OK;
}

/** @brief The -TY / -TEEN blur: "SIXTEE" may be SIXTY or SIXTEEN (mathspeech._doubt). */
static int blurred_word(const edu_math_word_t *w, int *ty, int *teen)
{
    char buffer[EDU_MAX_MATCH_LEN + 4U];
    const uint32_t n = w->length;
    uint32_t stem;

    if ((n < 5U) || (memcmp(&w->text[n - 3U], "TEE", 3U) != 0))
    {
        return 0;
    }
    stem = n - 3U;
    if (stem + 4U > EDU_MAX_MATCH_LEN)
    {
        return 0; /* too long to be a known word either way */
    }
    (void)memcpy(buffer, w->text, stem);
    (void)memcpy(&buffer[stem], "TEEN", 4U);
    *teen = canonical(buffer, stem + 4U);
    (void)memcpy(&buffer[stem], "TY", 2U);
    *ty = canonical(buffer, stem + 2U);
    return (*ty >= 0) && (word_class(*ty) == EDU_CLASS_TENS) && !edu_known[*ty].plural && (*teen >= 0) &&
           (word_class(*teen) == EDU_CLASS_TEEN) && !edu_known[*teen].plural;
}

/** @brief Append text at *used, keeping one byte for the terminator. */
static void append_text(char *buffer, uint32_t size, uint32_t *used, const char *text)
{
    while ((*text != '\0') && (*used + 1U < size))
    {
        buffer[(*used)++] = *text++;
    }
    buffer[*used] = '\0';
}

/** @brief Append a number, or @p none for EDU_MATH_NONE. */
static void append_number(char *buffer, uint32_t size, uint32_t *used, int32_t value, const char *none)
{
    char digits[8];
    char text[8];
    uint32_t count = 0U;
    uint32_t k = 0U;

    if (value < 0)
    {
        append_text(buffer, size, used, none);
        return;
    }
    do
    {
        digits[count++] = (char)('0' + (value % 10));
        value /= 10;
    } while ((value > 0) && (count < sizeof(digits) - 1U));
    while (count > 0U)
    {
        text[k++] = digits[--count];
    }
    text[k] = '\0';
    append_text(buffer, size, used, text);
}

/** @brief Python Claim.equation() with x for the multiplication sign; "" for NONE. */
static void append_equation(char *buffer, uint32_t size, uint32_t *used, const edu_math_claim_t *claim)
{
    if (claim->template_id == EDU_MATH_TEMPLATE_NONE)
    {
        return;
    }
    append_number(buffer, size, used, claim->a, "?");
    append_text(buffer, size, used, " x ");
    append_number(buffer, size, used, claim->b, "?");
    append_text(buffer, size, used, " = ");
    append_number(buffer, size, used, claim->c, "?");
}

/** @brief Both readings of a blurred word, "4 x 15 = 60 or 16" (parse()'s claim.shown). */
static void make_shown(edu_math_result_t *result)
{
    const edu_math_claim_t *claims = result->readings;
    const uint32_t count = result->reading_count;
    char *shown = result->shown;
    uint32_t used = 0U;
    uint32_t i;
    uint32_t j;
    int same_operands = (result->claim.a != EDU_MATH_NONE) && (result->claim.b != EDU_MATH_NONE);

    shown[0] = '\0';
    for (i = 1U; same_operands && (i < count); ++i)
    {
        same_operands = (claims[i].a == claims[0].a) && (claims[i].b == claims[0].b);
    }
    if (same_operands)
    {
        append_number(shown, EDU_MATH_SHOWN_CAPACITY, &used, result->claim.a, "?");
        append_text(shown, EDU_MATH_SHOWN_CAPACITY, &used, " x ");
        append_number(shown, EDU_MATH_SHOWN_CAPACITY, &used, result->claim.b, "?");
        append_text(shown, EDU_MATH_SHOWN_CAPACITY, &used, " = ");
        for (i = 0U; i < count; ++i)
        {
            int seen = 0;

            for (j = 0U; j < i; ++j)
            {
                seen = seen || (claims[j].c == claims[i].c);
            }
            if (seen)
            {
                continue;
            }
            if (i > 0U)
            {
                append_text(shown, EDU_MATH_SHOWN_CAPACITY, &used, " or ");
            }
            append_number(shown, EDU_MATH_SHOWN_CAPACITY, &used, claims[i].c, "None");
        }
        return;
    }
    for (i = 0U; i < count; ++i)
    {
        int seen = (claims[i].template_id == EDU_MATH_TEMPLATE_NONE);

        for (j = 0U; !seen && (j < i); ++j)
        {
            seen = (claims[j].template_id != EDU_MATH_TEMPLATE_NONE) && (claims[j].a == claims[i].a) &&
                   (claims[j].b == claims[i].b) && (claims[j].c == claims[i].c);
        }
        if (seen)
        {
            continue;
        }
        if (used > 0U)
        {
            append_text(shown, EDU_MATH_SHOWN_CAPACITY, &used, " or ");
        }
        append_equation(shown, EDU_MATH_SHOWN_CAPACITY, &used, &claims[i]);
    }
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

const edu_math_ranges_t *edu_math_default_ranges(void)
{
    return &default_ranges;
}

edu_math_status_t edu_math_parse(edu_math_work_t *work,
                                 const char *text,
                                 const int16_t *expect,
                                 const edu_math_ranges_t *ranges,
                                 edu_math_result_t *result)
{
    uint32_t positions[2];
    uint32_t i;
    uint32_t choice;
    edu_math_status_t status;

    if (result == NULL)
    {
        return EDU_MATH_ERR_ARGUMENT;
    }
    (void)memset(result, 0, sizeof(*result));
    none_claim(&result->claim);
    if ((work == NULL) || (text == NULL))
    {
        return EDU_MATH_ERR_ARGUMENT;
    }
    if (ranges == NULL)
    {
        ranges = &default_ranges;
    }
    status = normalize(work, text);
    if (status != EDU_MATH_OK)
    {
        return status;
    }
    for (i = 0U; (i < work->word_count) && (result->blurred < 2U); ++i)
    {
        int ty;
        int teen;

        if (blurred_word(&work->words[i], &ty, &teen))
        {
            positions[result->blurred] = i;
            result->blur_ty[result->blurred] = edu_known_words[ty];
            result->blur_teen[result->blurred] = edu_known_words[teen];
            ++result->blurred;
        }
    }
    if (result->blurred == 0U)
    {
        result->reading_count = 1U;
        status = parse_once(work, expect, ranges, &result->claim);
        result->readings[0] = result->claim;
        return status;
    }

    /* Read each blurred word both ways; neither a tick nor a cross is earned from a blurred word. */
    result->reading_count = (uint8_t)(1U << result->blurred);
    for (choice = 0U; choice < result->reading_count; ++choice)
    {
        edu_math_word_t saved[2];

        for (i = 0U; i < result->blurred; ++i)
        {
            const char *word = ((choice >> i) & 1U) ? result->blur_teen[i] : result->blur_ty[i];

            saved[i] = work->words[positions[i]];
            work->words[positions[i]].text = word;
            work->words[positions[i]].length = (uint8_t)strlen(word);
        }
        status = parse_once(work, expect, ranges, &result->readings[choice]);
        for (i = 0U; i < result->blurred; ++i)
        {
            work->words[positions[i]] = saved[i];
        }
        if (status != EDU_MATH_OK)
        {
            none_claim(&result->claim);
            return status;
        }
    }
    result->claim = result->readings[0];
    for (choice = 1U; choice < result->reading_count; ++choice)
    {
        if (result->readings[choice].correct != result->readings[0].correct)
        {
            result->doubt = 1U;
        }
    }
    if (result->doubt)
    {
        result->claim.correct = EDU_MATH_UNKNOWN;
        make_shown(result);
    }
    return EDU_MATH_OK;
}

edu_math_status_t edu_math_parse_many(edu_math_work_t *work,
                                      const char *text,
                                      int16_t table,
                                      const edu_math_ranges_t *ranges,
                                      edu_math_claim_t *claims,
                                      uint32_t capacity,
                                      uint32_t *claim_count,
                                      uint32_t *item_count)
{
    edu_math_status_t status;
    uint32_t found = 0U;
    uint32_t i;
    uint32_t j;

    if (claim_count != NULL)
    {
        *claim_count = 0U;
    }
    if (item_count != NULL)
    {
        *item_count = 0U;
    }
    if ((work == NULL) || (text == NULL) || (claim_count == NULL) || ((claims == NULL) && (capacity > 0U)))
    {
        return EDU_MATH_ERR_ARGUMENT;
    }
    if (ranges == NULL)
    {
        ranges = &default_ranges;
    }
    status = normalize(work, text);
    if (status == EDU_MATH_OK)
    {
        status = tokenize(work);
    }
    if (status != EDU_MATH_OK)
    {
        return status;
    }
    clean(work, 0U);
    if (item_count != NULL)
    {
        *item_count = work->item_count;
    }
    many_search(work, NULL, ranges, table);
    for (j = work->item_count; j > 0U; j = work->dp_prev[j])
    {
        found += work->dp_has[j];
    }
    /* Fill from the right so the claims come out in order. */
    i = found;
    for (j = work->item_count; j > 0U; j = work->dp_prev[j])
    {
        if (!work->dp_has[j])
        {
            continue;
        }
        --i;
        if (i < capacity)
        {
            many_claim(work, j, NULL, ranges, &claims[i]);
        }
    }
    *claim_count = found;
    return EDU_MATH_OK;
}

int edu_math_may_continue(edu_math_work_t *work, const char *text)
{
    const edu_math_word_t *last;
    uint32_t k;

    if ((work == NULL) || (text == NULL) || (normalize(work, text) != EDU_MATH_OK) || (work->word_count == 0U))
    {
        return 0;
    }
    last = &work->words[work->word_count - 1U];
    if (word_is(last->text, last->length, "AND") || word_is(last->text, last->length, "ND") ||
        word_is(last->text, last->length, "AN") || word_is(last->text, last->length, "A"))
    {
        return 1;
    }
    if (tokenize(work) != EDU_MATH_OK)
    {
        return 0;
    }
    for (k = work->token_count; k > 0U; --k)
    {
        const edu_math_token_t *t = &work->tokens[k - 1U];

        if ((t->kind == KIND_SKIP) || (t->kind == KIND_UNKNOWN))
        {
            continue;
        }
        return ((t->kind == KIND_HUNDRED) || (t->kind == KIND_TENS) || (t->kind == KIND_OP) ||
                (t->kind == KIND_EQ) || (t->kind == KIND_A)) && !t->plural;
    }
    return 0;
}

void edu_math_format(const edu_math_claim_t *claim, char *buffer, uint32_t size)
{
    uint32_t used = 0U;

    if ((buffer == NULL) || (size == 0U))
    {
        return;
    }
    buffer[0] = '\0';
    if (claim != NULL)
    {
        append_equation(buffer, size, &used, claim);
    }
}
