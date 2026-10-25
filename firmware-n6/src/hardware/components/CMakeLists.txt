# SPDX-FileCopyrightText: 2026 Nikita Prismakov
# SPDX-License-Identifier: Apache-2.0

# Drivers for external ICs (ST BSP "Components"), unmodified; the *_conf.h next
# to each is the project configuration. Used by the ST BSP XSPI driver.

# MX66UW1G45G octo-SPI NOR flash (XSPI2).
add_library(mx66uw1g45g mx66uw1g45g/mx66uw1g45g.c)
target_include_directories(mx66uw1g45g PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mx66uw1g45g)
target_link_libraries(mx66uw1g45g PUBLIC stm32n6_hal)

# APS256XX hexa-deca-SPI PSRAM (XSPI1).
add_library(aps256xx aps256xx/aps256xx.c)
target_include_directories(aps256xx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/aps256xx)
target_link_libraries(aps256xx PUBLIC stm32n6_hal)

# GT911 touch controller (I2C2) and the ST BSP's common touch interface (common/ts.h),
# unmodified; gt911/gt911_conf.h is the project configuration. EDU and translator modes only.
if(APP_EDU_MODE OR APP_TRANSLATOR_MODE)
    add_library(gt911 gt911/gt911.c gt911/gt911_reg.c)
    target_include_directories(gt911 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/gt911
        ${CMAKE_CURRENT_SOURCE_DIR}/common)
endif()

# WM8904 audio codec (I2C2, the DK's 3.5 mm jack), from github.com/STMicroelectronics/stm32-wm8904
# V1.1.1, unmodified. Translator mode only: the Hindi voice and the WBA music go out on the jack.
if(APP_TRANSLATOR_MODE)
    add_library(wm8904 wm8904/wm8904.c wm8904/wm8904_reg.c)
    target_include_directories(wm8904 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/wm8904)
endif()
