# SPDX-License-Identifier: Apache-2.0

# Translator mode (-DAPP_TRANSLATOR_MODE=ON): English speech in, Hindi out on a MAX98357A
# speaker that the N6 shares with a STM32WBA55 Bluetooth music receiver through a TMUX1574.
#   xl_audio_out.c   SAI2 block B I2S master (A2/D6/D8) fed from a sample ring by GPDMA1 ch. 7
#   xl_router.c      which board has the speaker: TMUX SEL (D7), amp shutdown (D4), music in (D2)
#   xl_display.c     the screen, the touch buttons and the power states; runs on the display thread
add_library(app_translator
    xl_audio_out.c
    xl_router.c
    xl_display.c)
target_include_directories(app_translator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app_translator PUBLIC mw_lcd threadx)
target_link_libraries(app_translator PRIVATE
    app_config app_audio app_text_case asr_engine boards bsp bsp_touch logger mcu_clock stm32n6_hal)
target_compile_options(app_translator PRIVATE -Wextra)
