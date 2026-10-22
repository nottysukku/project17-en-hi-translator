# SPDX-License-Identifier: Apache-2.0

# EDU mode (-DAPP_EDU_MODE=ON): the times-table quiz on the LCD.
#   edu_math.c       checks spoken multiplication; a C port of ASR_Math_Game/mathspeech.py,
#                    word tables generated into edu_math_tables.h (tools/gen_edu_tables.py),
#                    agreement with Python checked on the PC (test/run_host_test.py)
#   edu_game.c       the quiz: question, listen, tick or cross, next question; plain C
#   edu_display.c    the screens, the touch screen and the USER button; runs on the display thread
add_library(app_edu
    edu_math.c
    edu_game.c
    edu_display.c)
target_include_directories(app_edu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app_edu PUBLIC mw_lcd threadx)
target_link_libraries(app_edu PRIVATE app_config boards bsp bsp_touch logger)
# EDU builds write every function's stack frame (*.su, top-level CMakeLists.txt);
# tools/stack_report.py adds up the deepest call chain of the display thread.
target_compile_options(app_edu PRIVATE -Wextra)
