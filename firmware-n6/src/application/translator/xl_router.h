/**
 * @file     xl_router.h
 * @brief    What the 3.5 mm jack plays: the N6 translator's Hindi voice or the WBA's music.
 * @details  Both reach the one speaker (DK jack -> PAM8403 -> 4 Ohm 3 W): the WBA sends its music
 *           as I2S to the DK (xl_audio_out.c), and the choice is made in the DK's audio stream.
 *           Hardware (Arduino header of the STM32N6570-DK):
 *           - D7 = PD6, output: TMUX1574 EN (active low), with 10 kOhm to 3.3 V so the switch is
 *             open until this firmware has set its receive pins. Low connects the WBA's I2S
 *             (A2/D6/D8) and music line (D2) to the DK. The TMUX's SEL is tied to GND (side A).
 *           - D2 = PD0, input with pull-down (and 100 kOhm to GND): the WBA's "music playing" line.
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

/** @brief Which board the speaker plays. */
typedef enum
{
    XL_SOURCE_WBA = 0, /**< The WBA's music, passed through the DK. */
    XL_SOURCE_N6       /**< The translator's voice. */
} xl_source_t;

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

/** @brief Configure the two pins; the TMUX stays open, the music has the speaker (as without the N6). */
void xl_router_init(void);

/** @brief Close (non-zero) or open the TMUX1574 between the WBA and the DK. */
void xl_router_connect_wba(int connect);

/** @brief Choose the mode; takes effect on the next xl_router_poll(). */
void xl_router_set_mode(xl_mode_t mode);

/** @brief The mode chosen. */
xl_mode_t xl_router_mode(void);

/** @brief Non-zero while the WBA reports music (debounced: on after 100 ms, off after 1.5 s). */
int xl_router_music(void);

/** @brief The source the speaker plays now. */
xl_source_t xl_router_source(void);

/** @brief The source the rules want; differs from xl_router_source() during a switch. */
xl_source_t xl_router_target(void);

/**
 * @brief Read the music line, apply the rules, ask the audio stream for the source.
 * @param[in] now_ms Milliseconds since start-up.
 * @return Non-zero when the source or the music state changed in this call.
 * @note Call every few tens of milliseconds; a switch takes about 10 ms (fade out, fade in).
 */
int xl_router_poll(uint32_t now_ms);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_TRANSLATOR_XL_ROUTER_H */
