# CynipsNet v1.0 rebuilt: the weights extracted from cynipsnet_v1.0/network_atonbuf.xSPI2.bin,
# re-exported as QDQ ONNX and recompiled with ST Edge AI Core 4.0.1 (LL_ATON 1.1.3-275), the
# same memory pools as cynipsnet_v1.0. The generated code is compiled here from source, so it
# needs the matching runtime: configure with -DASR_MODEL_NETWORK=cynipsnet_v1.0_rt
# -DSTEDGEAI_ROOT=<ST Edge AI 4.0>/Middlewares/ST/AI. Weights are flashed separately.
add_library(asr_network STATIC network.c stai_network.c)
target_include_directories(asr_network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(asr_network PUBLIC stedgeai)
# Generated code: optimized like the vendor runtime, its warnings are ST's.
target_compile_options(asr_network PRIVATE -O2 -w)
