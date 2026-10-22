# NUCLEO-WBA55CG Bluetooth LE Audio music receiver for Project 17.
#
# Based on ST's BLE_Audio_TMAP_Peripheral (STM32WBA55G-DK1). The application sources live here;
# the HAL, the Bluetooth stack, LE Audio and LC3 libraries come from an STM32CubeWBA checkout
# (CUBE_WBA). BOARD selects the original Discovery kit build or the Nucleo port.
#
#   cmake -G Ninja -B build -DCMAKE_TOOLCHAIN_FILE=cmake/arm-gcc-cm33.cmake -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
cmake_minimum_required(VERSION 3.22)
project(wba_music_receiver C ASM)

set(CUBE_WBA "C:/Users/choprasu/STM32Cube/Repository/STM32CubeWBA_github" CACHE PATH "STM32CubeWBA root")
set(BOARD "NUCLEO" CACHE STRING "NUCLEO (Project 17) or DK1 (ST's original)")
set_property(CACHE BOARD PROPERTY STRINGS NUCLEO DK1)
if(NOT EXISTS "${CUBE_WBA}/Middlewares/ST/STM32_WPAN/ble/audio/lib/ble_audio.a")
    message(FATAL_ERROR "CUBE_WBA=${CUBE_WBA} has no LE Audio library: run 'git submodule update --init' there")
endif()

set(APP ${CMAKE_CURRENT_SOURCE_DIR})
set(WPAN ${CUBE_WBA}/Middlewares/ST/STM32_WPAN)
set(COMMON ${CUBE_WBA}/Projects/Common/WPAN)
set(HAL ${CUBE_WBA}/Drivers/STM32WBAxx_HAL_Driver)

set(HAL_MODULES
    hal hal_adc hal_adc_ex hal_cortex hal_crc hal_crc_ex hal_cryp hal_cryp_ex hal_dma hal_dma_ex
    hal_exti hal_flash hal_flash_ex hal_gpio hal_i2c hal_i2c_ex hal_pwr hal_pwr_ex hal_ramcfg
    hal_rcc hal_rcc_ex hal_rng hal_rng_ex hal_rtc hal_rtc_ex hal_sai hal_sai_ex hal_spi hal_spi_ex
    hal_uart hal_uart_ex ll_dma ll_exti ll_gpio ll_icache ll_pka ll_tim)
set(HAL_SOURCES)
foreach(m ${HAL_MODULES})
    list(APPEND HAL_SOURCES ${HAL}/Src/stm32wbaxx_${m}.c)
endforeach()

set(SOURCES
    ${HAL_SOURCES}
    ${CUBE_WBA}/Utilities/trace/adv_trace/stm32_adv_trace.c
    ${CUBE_WBA}/Utilities/lpm/tiny_lpm/stm32_lpm.c
    ${CUBE_WBA}/Utilities/misc/stm32_mem.c
    ${CUBE_WBA}/Utilities/sequencer/stm32_seq.c
    ${CUBE_WBA}/Utilities/misc/stm32_systime.c
    ${CUBE_WBA}/Utilities/tim_serv/stm32_timer.c
    ${CUBE_WBA}/Utilities/misc/stm32_tiny_sscanf.c
    ${CUBE_WBA}/Utilities/misc/stm32_tiny_vsnprintf.c
    ${COMMON}/Interfaces/adv_trace_usart_if.c
    ${COMMON}/Interfaces/app_bsp.c
    ${COMMON}/Interfaces/hw_aes.c
    ${COMMON}/Interfaces/hw_otp.c
    ${COMMON}/Interfaces/hw_rng.c
    ${COMMON}/Interfaces/timer_if.c
    ${COMMON}/Modules/app_sys.c
    ${COMMON}/Modules/crc_ctrl.c
    ${COMMON}/Modules/otp.c
    ${COMMON}/Modules/pka_ctrl.c
    ${COMMON}/Modules/scm.c
    ${COMMON}/Modules/stm_list.c
    ${COMMON}/Modules/stm_queue.c
    ${COMMON}/Modules/wpan_utils.c
    ${COMMON}/Modules/BasicAES/baes_ccm.c
    ${COMMON}/Modules/BasicAES/baes_cmac.c
    ${COMMON}/Modules/BasicAES/baes_ecb.c
    ${COMMON}/Modules/Flash/flash_driver.c
    ${COMMON}/Modules/Flash/flash_manager.c
    ${COMMON}/Modules/Flash/rf_timing_synchro.c
    ${COMMON}/Modules/Flash/simple_nvm_arbiter.c
    ${COMMON}/Modules/Log/log_module.c
    ${COMMON}/Modules/MemoryManager/advanced_memory_manager.c
    ${COMMON}/Modules/MemoryManager/stm32_mm.c
    ${COMMON}/Modules/RTDebug/RTDebug.c
    ${COMMON}/Modules/RTDebug/RTDebug_dtb.c
    ${COMMON}/Modules/SerialCmdInterpreter/serial_cmd_interpreter.c
    ${COMMON}/Startup/stm32wbaxx_ResetHandler_GCC.s
    ${WPAN}/ble/svc/Src/svc_ctl.c
    ${WPAN}/link_layer/ll_sys/src/ll_sys_cs.c
    ${WPAN}/link_layer/ll_sys/src/ll_sys_dp_slp.c
    ${WPAN}/link_layer/ll_sys/src/ll_sys_intf.c
    ${WPAN}/link_layer/ll_sys/src/ll_sys_startup.c
    ${APP}/gcc/startup_stm32wba55cgux.s
    ${APP}/gcc/syscalls.c
    ${APP}/gcc/sysmem.c
    ${APP}/Core/Src/system_stm32wbaxx.c
    ${APP}/Core/Src/app_entry.c
    ${APP}/Core/Src/main.c
    ${APP}/Core/Src/stm32wbaxx_hal_msp.c
    ${APP}/Core/Src/stm32wbaxx_it.c
    ${APP}/STM32_WPAN/App/app_ble.c
    ${APP}/STM32_WPAN/App/tmap_app.c
    ${APP}/STM32_WPAN/App/AudioUseCases/usecase_dev_mgmt.c
    ${APP}/STM32_WPAN/App/AudioUseCases/tmap/tmap.c
    ${APP}/STM32_WPAN/App/AudioUseCases/tmap/tmap_alloc.c
    ${APP}/STM32_WPAN/App/AudioUseCases/tmap/tmap_db.c
    ${APP}/STM32_WPAN/App/AudioUseCases/tmap/tmas.c
    ${APP}/STM32_WPAN/Target/ble_audio_if.c
    ${APP}/STM32_WPAN/Target/ble_audio_plat.c
    ${APP}/STM32_WPAN/Target/ble_codec.c
    ${APP}/STM32_WPAN/Target/bleplat.c
    ${APP}/STM32_WPAN/Target/codec_if.c
    ${APP}/STM32_WPAN/Target/host_stack_if.c
    ${APP}/STM32_WPAN/Target/linklayer_plat.c
    ${APP}/STM32_WPAN/Target/ll_sys_if.c
    ${APP}/STM32_WPAN/Target/power_table.c
    ${APP}/System/Interfaces/stm32_lpm_if.c
    ${APP}/System/Modules/ble_timer.c
    ${APP}/System/Config/CRC_Ctrl/crc_ctrl_conf.c
    ${APP}/System/Config/Debug_GPIO/app_debug.c
    ${APP}/System/Config/Flash/simple_nvm_arbiter_conf.c
    ${APP}/System/Config/LowPower/peripheral_init.c
    ${APP}/System/Config/LowPower/user_low_power_config.c)

set(INCLUDES
    ${APP}/Core/Inc
    ${APP}/System/Interfaces
    ${APP}/System/Modules
    ${APP}/System/Config
    ${APP}/System/Config/Log
    ${APP}/System/Config/LowPower
    ${APP}/System/Config/Debug_GPIO
    ${APP}/System/Config/Flash
    ${APP}/System/Config/CRC_Ctrl
    ${APP}/STM32_WPAN/App
    ${APP}/STM32_WPAN/Target
    ${APP}/STM32_WPAN/App/AudioUseCases
    ${APP}/STM32_WPAN/App/AudioUseCases/tmap
    ${HAL}/Inc
    ${HAL}/Inc/Legacy
    ${CUBE_WBA}/Utilities/trace/adv_trace
    ${COMMON}/Interfaces
    ${COMMON}/Modules
    ${COMMON}/Modules/BasicAES
    ${COMMON}/Modules/Flash
    ${COMMON}/Modules/MemoryManager
    ${COMMON}/Modules/Nvm
    ${COMMON}/Modules/RTDebug
    ${COMMON}/Modules/SerialCmdInterpreter
    ${COMMON}/Modules/Log
    ${CUBE_WBA}/Utilities/misc
    ${CUBE_WBA}/Utilities/sequencer
    ${CUBE_WBA}/Utilities/tim_serv
    ${CUBE_WBA}/Utilities/lpm/tiny_lpm
    ${WPAN}
    ${WPAN}/link_layer/ll_cmd_lib/config/ble_full
    ${WPAN}/ble/svc/Src
    ${CUBE_WBA}/Drivers/CMSIS/Device/ST/STM32WBAxx/Include
    ${WPAN}/link_layer/ll_cmd_lib/inc
    ${WPAN}/link_layer/ll_cmd_lib/inc/_40nm_reg_files
    ${WPAN}/link_layer/ll_cmd_lib/inc/ot_inc
    ${WPAN}/link_layer/ll_cmd_lib/porting
    ${WPAN}/link_layer/ll_sys/inc
    ${WPAN}/link_layer/ll_cmd_lib/src/shrd_utils/inc
    ${WPAN}/ble/stack/include
    ${WPAN}/ble/stack/include/auto
    ${WPAN}/ble/svc/Inc
    ${CUBE_WBA}/Drivers/CMSIS/Include
    ${CUBE_WBA}/Drivers/BSP/Components/Common
    ${WPAN}/ble/audio/Inc
    ${WPAN}/ble/codec/lc3/Inc
    ${WPAN}/ble/codec/codec_manager/Inc)

set(DEFINES USE_FULL_LL_DRIVER USE_HAL_DRIVER STM32WBA55xx PHY_40nm_3_00_a BLE_LL)

if(BOARD STREQUAL "DK1")
    list(APPEND SOURCES
        ${CUBE_WBA}/Utilities/LCD/stm32_lcd.c
        ${CUBE_WBA}/Drivers/BSP/STM32WBA55G-DK1/stm32wba55g_discovery.c
        ${CUBE_WBA}/Drivers/BSP/STM32WBA55G-DK1/stm32wba55g_discovery_audio.c
        ${CUBE_WBA}/Drivers/BSP/STM32WBA55G-DK1/stm32wba55g_discovery_bus.c
        ${CUBE_WBA}/Drivers/BSP/STM32WBA55G-DK1/stm32wba55g_discovery_lcd.c
        ${CUBE_WBA}/Drivers/BSP/Components/cs42l51/cs42l51.c
        ${CUBE_WBA}/Drivers/BSP/Components/cs42l51/cs42l51_reg.c
        ${CUBE_WBA}/Drivers/BSP/Components/ssd1315/ssd1315.c
        ${CUBE_WBA}/Drivers/BSP/Components/ssd1315/ssd1315_reg.c
        ${APP}/Core/Src/image.c
        ${APP}/STM32_WPAN/App/app_menu.c
        ${APP}/STM32_WPAN/App/app_menu_cfg.c)
    list(APPEND INCLUDES
        ${CUBE_WBA}/Drivers/BSP/STM32WBA55G-DK1
        ${CUBE_WBA}/Drivers/BSP/Components/cs42l51
        ${CUBE_WBA}/Drivers/BSP/Components/ssd1315
        ${APP}/Core/Src/lcd
        ${CUBE_WBA}/Utilities/LCD)
else()
    # Nucleo: the MAX98357A needs no codec, and there is no screen or joystick.
    list(APPEND SOURCES ${APP}/Board/nucleo_audio.c ${APP}/Board/music_link.c ${APP}/Board/menu_stubs.c)
    list(APPEND INCLUDES ${APP}/Board ${APP}/Board/dk1_shim)
    list(APPEND DEFINES BOARD_NUCLEO_MAX98357A)
endif()

add_executable(wba_music_receiver ${SOURCES})
target_include_directories(wba_music_receiver PRIVATE ${INCLUDES})
target_compile_definitions(wba_music_receiver PRIVATE ${DEFINES})
target_link_options(wba_music_receiver PRIVATE
    -T${APP}/gcc/STM32WBA55CGUX_FLASH.ld
    -Wl,-Map=$<TARGET_FILE_DIR:wba_music_receiver>/wba_music_receiver.map
    -Wl,--print-memory-usage)
target_link_libraries(wba_music_receiver PRIVATE
    -Wl,--start-group
    ${WPAN}/link_layer/ll_cmd_lib/lib/LinkLayer_BLE_Full_V3_0.a
    ${WPAN}/ble/stack/lib/stm32wba_ble_stack_full.a
    ${WPAN}/ble/audio/lib/ble_audio.a
    ${WPAN}/ble/codec/codec_manager/Lib/codec_mngr.a
    ${WPAN}/ble/codec/lc3/Lib/LC3.a
    c m
    -Wl,--end-group)
set_target_properties(wba_music_receiver PROPERTIES SUFFIX ".elf")
add_custom_command(TARGET wba_music_receiver POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:wba_music_receiver> $<TARGET_FILE_DIR:wba_music_receiver>/wba_music_receiver.bin
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:wba_music_receiver>)
