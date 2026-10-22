/**
 * @file     xl_router.h
 * @brief    Which source the speaker plays: the N6 translator or the WBA Bluetooth music.
 * @details  Hardware (Arduino header of the STM32N6570-DK):
 *           - D7 = PD6, output: TMUX1574 SEL. Low = side A = WBA (also the hardware default, through
 *             a 100 kOhm pull-down, so music plays while the N6 sleeps); high = side B = N6.
 *           - D4 = PH5, open-drain output to the MAX98357A SD pin through 1 kOhm: driven low the
 *             amplifier is shut down (used for a few ms around every switch, no click); released,
 *             the module's own bias sets the (L+R)/2 mix.
 *           - D2 = PD0, input with pull-down: the WBA's "music playing" line.
 *           Rules: MUSIC mode plays the WBA; TRANSLATE mode plays the N6; AUTO mode (default) plays
 *           the WBA while music is playing and the N6 otherwise.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APPLICATION_TRANSLATOR_XL_ROUTER_H
#define APPLICATION_TRANSLATOR_XL_ROUTER_H

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************/
/*                                 PUBLIC TYPES                                */
/*******************************************************************************/

/** @brief What the user chose on the touch screen. */
typedef enum
{
    XL_MODE_AUTO = 0,  /**< Music when it plays, translation otherwise (default). */
    XL_MODE_TRANSLATE, /**< Always the N6 translation. */
    XL_MODE_MUSIC      /**< Always the WBA music. */
} xl_mode_t;

/** @brief Which board drives the speaker. */
typedef enum
{
    XL_SOURCE_WBA = 0, /**< TMUX side A, SEL low. */
    XL_SOURCE_N6       /**< TMUX side B, SEL high. */
} xl_source_t;

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/** @brief Configure the three pins; the speaker starts on the WBA (as without the N6). */
void xl_router_init(void);

/** @brief Choose the mode; takes effect on the next xl_router_poll(). */
void xl_router_set_mode(xl_mode_t mode);

/** @brief The mode chosen. */
xl_mode_t xl_router_mode(void);

/** @brief Non-zero while the WBA reports music (debounced: on after 100 ms, off after 1.5 s). */
int xl_router_music(void);

/** @brief The source the speaker is connected to now. */
xl_source_t xl_router_source(void);

/** @brief The source the rules want; differs from xl_router_source() during a switch. */
xl_source_t xl_router_target(void);

/**
 * @brief Read the music line, apply the rules, step a pending switch (mute, SEL, unmute).
 * @param[in] now_ms Milliseconds since start-up.
 * @return Non-zero when the source or the music state changed in this call.
 * @note Call every few tens of milliseconds; the switch takes about 50 ms.
 */
int xl_router_poll(uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_ROUTER_H */
