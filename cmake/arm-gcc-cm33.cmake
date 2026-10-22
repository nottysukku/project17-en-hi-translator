# Toolchain: GNU Arm (the copy inside STM32CubeIDE), Cortex-M33 with FPU, as in ST's project.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

file(GLOB _gnu_bins
    "C:/ST/STM32CubeIDE_*/STM32CubeIDE/plugins/*gnu-tools-for-stm32*/tools/bin"
    "$ENV{HOME}/st/stm32cubeide_*/plugins/*gnu-tools-for-stm32*/tools/bin"
    "/opt/st/stm32cubeide_*/plugins/*gnu-tools-for-stm32*/tools/bin")
find_program(CMAKE_C_COMPILER arm-none-eabi-gcc HINTS ${_gnu_bins} REQUIRED)
find_program(CMAKE_ASM_COMPILER arm-none-eabi-gcc HINTS ${_gnu_bins} REQUIRED)
find_program(CMAKE_OBJCOPY arm-none-eabi-objcopy HINTS ${_gnu_bins} REQUIRED)
find_program(CMAKE_SIZE arm-none-eabi-size HINTS ${_gnu_bins} REQUIRED)

set(_cpu "-mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard")
set(CMAKE_C_FLAGS_INIT "${_cpu} -std=gnu11 -ffunction-sections -fdata-sections -fstack-usage -Wall")
set(CMAKE_ASM_FLAGS_INIT "${_cpu} -x assembler-with-cpp")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_cpu} --specs=nano.specs --specs=nosys.specs -Wl,--gc-sections -z noexecstack")
set(CMAKE_C_FLAGS_RELEASE "-Os")
set(CMAKE_C_FLAGS_DEBUG "-Os -g3")
