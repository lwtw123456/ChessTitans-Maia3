cmake_minimum_required(VERSION 3.24)

project(maia_ai_dll LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(SAFETYHOOK_DIR  "${THIRD_PARTY_DIR}")
set(IMGUI_DIR       "${THIRD_PARTY_DIR}")
set(KIERO_DIR       "${THIRD_PARTY_DIR}")
set(ORT_ROOT        "${THIRD_PARTY_DIR}/onnxruntime-static")
set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
set(ORT_LIB_DIR     "${ORT_ROOT}/lib")

set(MAIA_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/maia3-5m.fp32.onnx")

foreach(required_file
    "${CMAKE_CURRENT_SOURCE_DIR}/maia_ai_dll.cpp"
    "${SAFETYHOOK_DIR}/safetyhook.cpp"
    "${SAFETYHOOK_DIR}/safetyhook.hpp"
    "${SAFETYHOOK_DIR}/Zydis.c"
    "${SAFETYHOOK_DIR}/Zydis.h"

    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui.h"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_dx9.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_dx9.h"

    "${KIERO_DIR}/kiero.hpp"
    "${KIERO_DIR}/kiero_intern.cpp"
    "${KIERO_DIR}/kiero_intern.hpp"
    "${KIERO_DIR}/kiero_d3d9.cpp"
    "${KIERO_DIR}/kiero_d3d9.hpp"

    "${ORT_INCLUDE_DIR}/onnxruntime/core/session/onnxruntime_cxx_api.h"
)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required file not found: ${required_file}")
    endif()
endforeach()

if(NOT IS_DIRECTORY "${ORT_LIB_DIR}")
    message(FATAL_ERROR "ONNX Runtime lib directory not found: ${ORT_LIB_DIR}")
endif()

set(ORT_CORE_LIBRARY_NAMES
    onnxruntime_session.lib
    onnxruntime_optimizer.lib
    onnxruntime_providers.lib
    onnxruntime_lora.lib
    onnxruntime_framework.lib
    onnxruntime_graph.lib
    onnxruntime_util.lib
    onnxruntime_mlas.lib
    onnxruntime_common.lib
    onnxruntime_flatbuffers.lib

    onnx.lib
    onnx_proto.lib
    libprotobuf.lib
    flatbuffers.lib
    cpuinfo.lib
)

set(ORT_CORE_LIBRARIES)

foreach(lib_name IN LISTS ORT_CORE_LIBRARY_NAMES)
    set(lib_path "${ORT_LIB_DIR}/${lib_name}")

    if(EXISTS "${lib_path}")
        list(APPEND ORT_CORE_LIBRARIES "${lib_path}")
    else()
        message(WARNING
            "ONNX Runtime dependency not present and will be skipped: ${lib_name}")
    endif()
endforeach()

file(GLOB ORT_ABSL_LIBRARIES CONFIGURE_DEPENDS
    "${ORT_LIB_DIR}/absl_*.lib")
list(SORT ORT_ABSL_LIBRARIES)

if(NOT ORT_ABSL_LIBRARIES)
    message(FATAL_ERROR "No absl_*.lib files found in ${ORT_LIB_DIR}")
endif()

add_library(maia_ai_dll SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/maia_ai_dll.cpp"

    "${SAFETYHOOK_DIR}/safetyhook.cpp"
    "${SAFETYHOOK_DIR}/safetyhook.hpp"

    "${SAFETYHOOK_DIR}/Zydis.c"
    "${SAFETYHOOK_DIR}/Zydis.h"

    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_dx9.cpp"

    "${KIERO_DIR}/kiero_intern.cpp"
    "${KIERO_DIR}/kiero_d3d9.cpp"
)

target_include_directories(maia_ai_dll PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${THIRD_PARTY_DIR}"
    "${ORT_INCLUDE_DIR}"
)

target_compile_definitions(maia_ai_dll PRIVATE
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    ZYDIS_STATIC_BUILD
)

target_link_libraries(maia_ai_dll PRIVATE
    ${ORT_CORE_LIBRARIES}
    ${ORT_ABSL_LIBRARIES}

    advapi32
    bcrypt
    dbghelp
    d3d9
    ole32
    oleaut32
    shell32
    shlwapi
    userenv
    uuid
    ws2_32
)

if(MSVC)
    target_compile_options(maia_ai_dll PRIVATE
        /MP
        /EHsc
        /utf-8
        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:/O1>
        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:/Gy>
        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:/Gw>
        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:/GF>
    )

    target_link_options(maia_ai_dll PRIVATE
        /INCREMENTAL:NO

        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:/OPT:REF>
        $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:/OPT:ICF>
    )
endif()

set_target_properties(maia_ai_dll PROPERTIES
    OUTPUT_NAME "slc"
)