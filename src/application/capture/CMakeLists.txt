# SPDX-License-Identifier: Apache-2.0

# Data-collection build only (-DAPP_AUDIO_CAPTURE=ON): microphone audio and log text as framed
# packets on the console UART. APP_AUDIO_CAPTURE reaches every target that links this one.
add_library(app_capture capture.c)
target_include_directories(app_capture PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(app_capture PUBLIC APP_AUDIO_CAPTURE=1)
target_link_libraries(app_capture PRIVATE app_config drv_uart threadx)
