/**
 * @file     edu_math.h
 * @brief    Check spoken multiplication: "TWENTY THREE TIMES ATE IS ONE EIGHTY FOR" -> 23 x 8 = 184, right.
 * @details  C99 port of ASR_Math_Game/mathspeech.py (parse, parse_many, may_continue), kept in step
 *           with it by a host test that must agree with Python on every test vector
 *           (test/run_host_test.py). The word lists are generated from the Python ones
 *           (tools/gen_edu_tables.py -> edu_math_tables.h).
 *
 *           Mapping words never looks at the expected answer, so a wrong answer is never corrected
 *           into a right one. A word the recognizer blurs between -TY and -TEEN ("SIXTEE") is read
 *           both ways; if the verdicts differ the result asks the child to say it again.
 *
 *           No heap, no recursion deeper than four frames: all working memory is in an
 *           edu_math_work_t the caller owns (about 7 KB on the board; keep it static). Not re-entrant per work
 *           area. Input is UTF-8 text; the recognizer's own output is A-Z, apostrophe, space and
 *           , . ? marks.
 *
 *           Differences from Python, outside what the board can produce: digit strings of 1000 or
 *           more are dropped (Python raises), non-ASCII characters other than the multiplication
 *           sign and the right single quote separate words (Python upper-cases a few into ASCII
 *           letters), and text beyond the limits below returns EDU_MATH_ERR_LIMIT.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_EDU_EDU_MATH_H
#define APPLICATION_EDU_EDU_MATH_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                                PUBLIC DEFINES                               */
/*******************************************************************************/

/** @brief A number that was not said (Python None). */
#define EDU_MATH_NONE (-1)

/** @brief edu_math_claim_t.correct when the claim cannot be marked (Python None). */
#define EDU_MATH_UNKNOWN (-1)

/** @brief Characters of normalized text (words plus one separator each, digits spelt out). */
#define EDU_MATH_TEXT_CAPACITY 1024U

/** @brief Words after normalization. */
#define EDU_MATH_MAX_WORDS 192U

/** @brief Letters in one word; a longer one gives EDU_MATH_ERR_LIMIT. */
#define EDU_MATH_MAX_WORD_LEN 64U

/** @brief Tokens after glued words are split; also the bound on cleaned tokens (at most 255). */
#define EDU_MATH_MAX_TOKENS 240U

/** @brief Room in edu_math_result_t.shown, terminator included. */
#define EDU_MATH_SHOWN_CAPACITY 96U

/*******************************************************************************/
/*                                 PUBLIC TYPES                                */
/*******************************************************************************/

/** @brief Result of a call. */
typedef enum
{
    EDU_MATH_OK = 0,        /**< Parsed (the claim may still be EDU_MATH_TEMPLATE_NONE). */
    EDU_MATH_ERR_ARGUMENT,  /**< A required pointer was NULL. */
    EDU_MATH_ERR_LIMIT      /**< The text exceeds a limit above; the claim is NONE. */
} edu_math_status_t;

/** @brief What the child said. */
typedef enum
{
    EDU_MATH_TEMPLATE_NONE = 0, /**< No sum could be read. */
    EDU_MATH_TEMPLATE_FULL,     /**< A x B = C. */
    EDU_MATH_TEMPLATE_ANSWER,   /**< C only (checked against the question). */
    EDU_MATH_TEMPLATE_PARTIAL,  /**< B = C ("EIGHTS ARE ..."). */
    EDU_MATH_TEMPLATE_QUESTION  /**< A x B, no answer yet. */
} edu_math_template_t;

/** @brief Allowed operands, as mathspeech.Ranges: a table_min..table_max times mult_min..mult_max line. */
typedef struct
{
    int16_t table_min; /**< Default 2. */
    int16_t table_max; /**< Default 30. */
    int16_t mult_min;  /**< Default 1. */
    int16_t mult_max;  /**< Default 15. */
} edu_math_ranges_t;

/** @brief One reading of an utterance (mathspeech.Claim without the word report and notes). */
typedef struct
{
    uint8_t template_id; /**< edu_math_template_t. */
    int8_t correct;      /**< 1 right, 0 wrong, EDU_MATH_UNKNOWN cannot check. */
    uint8_t ambiguous;   /**< 1 when the same words also read as another sum just as well. */
    uint8_t span_first;  /**< First cleaned token of the claim (parse_many). */
    uint8_t span_end;    /**< One past its last cleaned token. */
    int16_t a;           /**< First operand, or EDU_MATH_NONE. */
    int16_t b;           /**< Second operand, or EDU_MATH_NONE. */
    int16_t c;           /**< The answer said, or EDU_MATH_NONE. */
    int16_t score;       /**< Score of the chosen reading (higher is likelier). */
} edu_math_claim_t;

/** @brief Result of edu_math_parse(). */
typedef struct
{
    /** @brief What to mark. With doubt set, correct is EDU_MATH_UNKNOWN. */
    edu_math_claim_t claim;
    /** @brief 1: a blurred -ty/-teen word changes the verdict; neither tick nor cross: ask again. */
    uint8_t doubt;
    /** @brief Blurred words read both ways (0..2), in order of appearance. */
    uint8_t blurred;
    /** @brief Readings tried: 1 << blurred (claim is readings[0] when blurred > 0). */
    uint8_t reading_count;
    /** @brief The -TY reading of each blurred word ("SIXTY"). */
    const char *blur_ty[2];
    /** @brief The -TEEN reading of each blurred word ("SIXTEEN"). */
    const char *blur_teen[2];
    /** @brief Claim of every reading, choice bit k = TEEN for blurred word k. */
    edu_math_claim_t readings[4];
    /** @brief With doubt: both readings, "4 x 15 = 60 or 16" (Python Claim.shown); else empty. */
    char shown[EDU_MATH_SHOWN_CAPACITY];
} edu_math_result_t;

/** @brief One heard word (internal). */
typedef struct
{
    const char *text; /**< Upper-case letters, not terminated. */
    uint8_t length;   /**< Letters. */
} edu_math_word_t;

/** @brief One word as mapped (internal): known-word index or -1, and how it was found. */
typedef struct
{
    int16_t hit;      /**< Index into the known words, or -1. */
    uint8_t fuzzy;    /**< 1 when matched by edit distance. */
    uint8_t heard_to; /**< 1 when the heard word was exactly "TO". */
} edu_math_entry_t;

/** @brief One token (internal): kind, number value, plural flag. */
typedef struct
{
    uint8_t kind;     /**< Token kind (edu_math.c). */
    uint8_t plural;   /**< 1 for TWOS, EIGHTS, ... */
    uint8_t heard_to; /**< 1 when heard as "TO". */
    uint8_t value;    /**< Number value; HUNDRED is 100. */
} edu_math_token_t;

/** @brief One number of a reading (internal). */
typedef struct
{
    int16_t value;  /**< The number. */
    uint8_t style;  /**< Standard, colloquial ("ONE EIGHTY FOUR") or digits ("ONE EIGHT FOUR"). */
    uint8_t plural; /**< Its last word is a plural. */
    uint8_t first;  /**< First token. */
    uint8_t last;   /**< Last token. */
} edu_math_num_t;

/** @brief One cell of the glued-word split (internal). */
typedef struct
{
    int16_t cost;  /**< Twice the parts, plus one per fuzzy part. */
    uint8_t prev;  /**< Where the last part starts. */
    uint8_t set;   /**< 1 once reachable. */
    int16_t hit;   /**< Known word of the last part. */
    uint8_t fuzzy; /**< The last part was a fuzzy match. */
} edu_math_split_t;

/**
 * @brief Working memory of the checker; owned by the caller, contents private.
 * @note Keep it static: it is too large for a thread stack.
 */
typedef struct
{
    char text[EDU_MATH_TEXT_CAPACITY];
    edu_math_word_t words[EDU_MATH_MAX_WORDS];
    uint32_t word_count;
    edu_math_entry_t entries[EDU_MATH_MAX_TOKENS];
    uint32_t entry_count;
    edu_math_token_t tokens[EDU_MATH_MAX_TOKENS];
    uint32_t token_count;
    edu_math_token_t items[EDU_MATH_MAX_TOKENS];
    uint32_t item_count;
    edu_math_split_t split[EDU_MATH_MAX_WORD_LEN + 1U];
    /* reading search (_best) */
    const edu_math_token_t *e_items;
    uint32_t e_count;
    const int16_t *e_expect;
    const edu_math_ranges_t *e_ranges;
    uint8_t e_templates;
    uint8_t e_pass;
    uint8_t e_found;
    uint8_t e_ambiguous;
    edu_math_num_t e_nums[3];
    edu_math_num_t e_best_nums[3];
    uint8_t e_best_template;
    uint8_t e_best_count;
    int16_t e_best_total;
    int16_t e_best_base;
    /* sequence of claims (_parse_items_many) */
    int32_t dp_score[EDU_MATH_MAX_TOKENS + 1U];
    uint8_t dp_prev[EDU_MATH_MAX_TOKENS + 1U];
    uint8_t dp_has[EDU_MATH_MAX_TOKENS + 1U];
} edu_math_work_t;

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/**
 * @brief The default ranges: tables 2-30, multipliers 1-15.
 * @return A pointer valid for the life of the program.
 */
const edu_math_ranges_t *edu_math_default_ranges(void);

/**
 * @brief Read one utterance (mathspeech.parse).
 * @param[in,out] work Working memory.
 * @param[in] text What the recognizer heard, NUL-terminated UTF-8.
 * @param[in] expect The question on screen {a, b} (quiz mode), or NULL (free mode).
 * @param[in] ranges Allowed operands, or NULL for the defaults.
 * @param[out] result The reading and its verdict.
 * @return EDU_MATH_OK, or an error with result->claim NONE.
 */
edu_math_status_t edu_math_parse(edu_math_work_t *work,
                                 const char *text,
                                 const int16_t *expect,
                                 const edu_math_ranges_t *ranges,
                                 edu_math_result_t *result);

/**
 * @brief Read a whole recital as a sequence of A x B = C claims (mathspeech.parse_many).
 * @param[in,out] work Working memory.
 * @param[in] text What the recognizer heard.
 * @param[in] table The table being recited (its lines are preferred), or EDU_MATH_NONE.
 * @param[in] ranges Allowed operands, or NULL for the defaults.
 * @param[out] claims Room for the claims, in order.
 * @param[in] capacity Entries in @p claims; further claims are counted but not stored.
 * @param[out] claim_count Claims found.
 * @param[out] item_count Cleaned tokens (numbers and markers) heard; may be NULL.
 * @return EDU_MATH_OK, or an error with no claims.
 */
edu_math_status_t edu_math_parse_many(edu_math_work_t *work,
                                      const char *text,
                                      int16_t table,
                                      const edu_math_ranges_t *ranges,
                                      edu_math_claim_t *claims,
                                      uint32_t capacity,
                                      uint32_t *claim_count,
                                      uint32_t *item_count);

/**
 * @brief Tell whether the words so far expect more ("... ONE HUNDRED", "... EIGHTY", "... IS", "... AND").
 * @details The game then waits a little longer before judging (mathspeech.may_continue).
 * @return 1 or 0 (0 also on error).
 */
int edu_math_may_continue(edu_math_work_t *work, const char *text);

/**
 * @brief Write a claim as "23 x 8 = 184" ("?" for a number not said; empty for NONE).
 * @param[in] claim The claim.
 * @param[out] buffer Destination, always terminated when size > 0.
 * @param[in] size Bytes in @p buffer.
 */
void edu_math_format(const edu_math_claim_t *claim, char *buffer, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_EDU_EDU_MATH_H */
