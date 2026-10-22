# SPDX-License-Identifier: Apache-2.0

# English-to-Hindi translation engine (IndicTrans2, it2.c): portable C, Helium on the M55.
# Always optimized: Debug builds at -O0 would make a translation many times slower.
add_library(it2 STATIC it2.c)
target_include_directories(it2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(it2 PRIVATE -O2 -Wextra)
