/**
 * @file     text_case.c
 * @brief    Sentence case for the recognized text; implements text_case.h.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "text_case.h"

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Non-zero for an ASCII letter of either case. */
static int is_letter(char character)
{
    return ((character >= 'A') && (character <= 'Z')) || ((character >= 'a') && (character <= 'z'));
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void text_sentence_case(char *text, uint32_t length, int sentence_start)
{
    int starts_sentence = sentence_start;
    uint32_t index;

    for (index = 0U; index < length; ++index)
    {
        const char character = text[index];

        if (is_letter(character))
        {
            /* The word I: a lone letter I, or one followed by an apostrophe (I'M). */
            const int word_start = (index == 0U) ||
                                   !(is_letter(text[index - 1U]) || (text[index - 1U] == '\''));
            const int word_end = ((index + 1U) == length) || !is_letter(text[index + 1U]);
            const int pronoun = word_start && word_end && ((character == 'I') || (character == 'i'));
            const int upper = starts_sentence || pronoun;

            if (upper && (character >= 'a'))
            {
                text[index] = (char)(character - ('a' - 'A'));
            }
            else if (!upper && (character <= 'Z'))
            {
                text[index] = (char)(character + ('a' - 'A'));
            }
            starts_sentence = 0;
        }
        else if ((character == '.') || (character == '?'))
        {
            starts_sentence = 1;
        }
    }
}
