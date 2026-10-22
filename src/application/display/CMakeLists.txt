# SPDX-FileCopyrightText: 2026 Nikita Prismakov
# SPDX-License-Identifier: Apache-2.0

# LCD: the startup logo, then the transcript the ASR model recognizes.
#   display_thread.c  the thread, its mailbox and the asr_engine text port
#   transcript.c      word-wrapped text on the screen; plain C, no RTOS
#   text_case.c       sentence case for models trained with punctuation (its own library: the
#                     translator screen uses it too)
#   startup_logo.c    the picture in images/ (tools/image_to_rgb565.py)
add_library(app_text_case text_case.c)
target_include_directories(app_text_case PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(app_display
    display_thread.c
    transcript.c
    startup_logo.c)
target_include_directories(app_display PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app_display PUBLIC asr_engine mw_lcd app_text_case)
target_link_libraries(app_display PRIVATE app_config bsp drv_ltdc logger threadx)
