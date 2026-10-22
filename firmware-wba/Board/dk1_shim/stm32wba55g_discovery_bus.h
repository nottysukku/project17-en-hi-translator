/**
 * @file     stm32wba55g_discovery_bus.h
 * @brief    NUCLEO-WBA55CG stand-in for the DK1 bus BSP: the I2C bus only served the CS42L51
 *           codec, which the MAX98357A does not need, so every call is a no-op.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef NUCLEO_SHIM_STM32WBA55G_DISCOVERY_BUS_H
#define NUCLEO_SHIM_STM32WBA55G_DISCOVERY_BUS_H

#include "stm32wba55g_discovery.h"

/* app_entry.c maps BSP_I2C_Init/DeInit to these; the I2C3 clock itself is harmless to toggle. */
static inline int32_t BSP_I2C3_Init(void) { return BSP_ERROR_NONE; }
static inline int32_t BSP_I2C3_DeInit(void) { return BSP_ERROR_NONE; }
static inline int32_t BSP_I2C3_WriteReg(uint16_t DevAddr, uint16_t Reg, uint8_t *pData, uint16_t Length)
{
  (void)DevAddr;
  (void)Reg;
  (void)pData;
  (void)Length;
  return BSP_ERROR_NONE;
}

/* Codec register names the application's microphone-gain helper still mentions. */
#define AUDIO_I2C_ADDRESS         0x94U
#define CS42L51_ADCA_ATTENUATOR   0x0CU
#define CS42L51_MIC_CTRL          0x07U

#endif /* NUCLEO_SHIM_STM32WBA55G_DISCOVERY_BUS_H */
