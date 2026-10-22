/**
 * @file     numbers_host.c
 * @brief    PC test of tts_numbers.c.
 *   cc -I.. ../tts_numbers.c numbers_host.c -o numbers_host && ./numbers_host
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <string.h>

#include "tts_numbers.h"

int main(void)
{
    static const char *const cases[][2] = {
        {"वह 25 साल की है।", "वह पच्चीस साल की है।"},
        {"मेरे पास 2,500 रुपये हैं", "मेरे पास दो हज़ार पाँच सौ रुपये हैं"},
        {"1,50,000", "एक लाख पचास हज़ार"},
        {"3.5 किलो", "तीन दशमलव पाँच किलो"},
        {"25% छूट", "पच्चीस प्रतिशत छूट"},
        {"साल 2024 में", "साल दो हज़ार चौबीस में"},
        {"100", "एक सौ"},
        {"0", "शून्य"},
        {"12,34,56,789", "बारह करोड़ चौंतीस लाख छप्पन हज़ार सात सौ नवासी"},
        {"9876543210", "नौ आठ सात छह पाँच चार तीन दो एक शून्य"},
        {"१२ बजे", "बारह बजे"},
        {"10मिनट", "दस मिनट"},
        {"कोई अंक नहीं", "कोई अंक नहीं"},
    };
    char out[512];
    unsigned i, failures = 0;

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
    {
        tts_spell_numbers(cases[i][0], out, sizeof(out));
        if (strcmp(out, cases[i][1]) != 0)
        {
            printf("FAIL: %s -> %s (expected %s)\n", cases[i][0], out, cases[i][1]);
            failures++;
        }
    }
    tts_spell_numbers("12345 12345 12345", out, 20); /* cut, still terminated */
    printf("cut to 20 bytes: %u bytes\n", (unsigned)strlen(out));
    printf(failures ? "%u FAILURES\n" : "all number tests passed\n", failures);
    return failures != 0;
}
