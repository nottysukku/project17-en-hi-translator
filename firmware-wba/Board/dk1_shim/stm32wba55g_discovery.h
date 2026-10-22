/**
 * @file     stm32wba55g_discovery.h
 * @brief    NUCLEO-WBA55CG stand-in for the STM32WBA55G-DK1 BSP header.
 * @details  ST's TMAP application includes the Discovery kit headers. On the Nucleo there is no
 *           joystick, screen or codec, so this directory replaces them with the few types and
 *           constants the application still names. The real audio output is Board/nucleo_audio.c.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef NUCLEO_SHIM_STM32WBA55G_DISCOVERY_H
#define NUCLEO_SHIM_STM32WBA55G_DISCOVERY_H

#include "stm32wbaxx_hal.h"

#define BSP_ERROR_NONE                    0
#define BSP_ERROR_NO_INIT                -1
#define BSP_ERROR_WRONG_PARAM            -2
#define BSP_ERROR_BUSY                   -3
#define BSP_ERROR_PERIPH_FAILURE         -4
#define BSP_ERROR_COMPONENT_FAILURE      -5
#define BSP_ERROR_UNKNOWN_FAILURE        -6
#define BSP_ERROR_UNKNOWN_COMPONENT      -7
#define BSP_ERROR_BUS_FAILURE            -8
#define BSP_ERROR_CLOCK_FAILURE          -9
#define BSP_ERROR_MSP_FAILURE            -10
#define BSP_ERROR_FEATURE_NOT_SUPPORTED  -11

/** @brief Joystick states; the application keeps one even without a joystick. */
typedef enum
{
  JOY_NONE  = 0x00U,
  JOY_SEL   = 0x01U,
  JOY_DOWN  = 0x02U,
  JOY_LEFT  = 0x04U,
  JOY_RIGHT = 0x08U,
  JOY_UP    = 0x10U,
  JOY_ALL   = 0x1FU
} JOYPin_TypeDef;

#endif /* NUCLEO_SHIM_STM32WBA55G_DISCOVERY_H */
