# SPDX-FileCopyrightText: 2026 Nikita Prismakov
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(middleware)
add_subdirectory(asr_engine)
add_subdirectory(hardware)
if(APP_TRANSLATOR_MODE)
    add_subdirectory(translate)
    add_subdirectory(tts)
endif()
add_subdirectory(application)
