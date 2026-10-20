# Original Xbox static recompilation: xbe_recomp, the host runtime and add_xbe_image()
include_guard(GLOBAL)

get_filename_component(XBE_SDK_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
# build scripts find the SDK's xdk folder and THIRD_PARTY.txt through this
file(TO_NATIVE_PATH "${XBE_SDK_DIR}" XBE_SDK_DIR_NATIVE)
file(WRITE ${CMAKE_BINARY_DIR}/sdk_dir.txt "${XBE_SDK_DIR_NATIVE}\n")


# clang targeting the MSVC ABI leaves CMAKE_<LANG>_FLAGS_<CONFIG> empty here, so a Release build
# compiled the host runtime, the dependencies and the tools at -O0 while only the generated guest
# code carried the -O2 set per source below. Measured consequence: the recorder's per-draw state
# compare cost several times what its byte count predicts. NDEBUG is deliberately left undefined -
# the log's verbose mode keys off it (core/log.cpp), and dropping to the release log format is a
# separate decision from optimising.
#
# Targeting Linux, clang does get CMake's own defaults, and those carry -DNDEBUG. That made the
# same Release build quiet on Linux and verbose on Windows: logs/<game>.log held the boot preamble
# and nothing after it, no rotation into <game>_001.log, and XBE_PERF / the present geometry lines
# the docs tell people to read produced nothing at all. The flags are stripped of NDEBUG here so
# one Release preset means one thing on both platforms.
foreach(cfg RELEASE RELWITHDEBINFO)
    foreach(lang C CXX)
        if(NOT CMAKE_${lang}_FLAGS_${cfg} MATCHES "[-/]O")
            set(CMAKE_${lang}_FLAGS_${cfg} "${CMAKE_${lang}_FLAGS_${cfg}} -O2")
        endif()
        string(REGEX REPLACE "[-/]D[ \t]*NDEBUG" "" stripped "${CMAKE_${lang}_FLAGS_${cfg}}")
        string(STRIP "${stripped}" stripped)
        set(CMAKE_${lang}_FLAGS_${cfg} "${stripped}" CACHE STRING "" FORCE)
    endforeach()
endforeach()
include(FetchContent)
set(ZYDIS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zydis GIT_REPOSITORY https://github.com/zyantific/zydis.git GIT_TAG v4.1.1 GIT_SHALLOW TRUE)
FetchContent_Declare(tomlplusplus GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git GIT_TAG v3.4.0 GIT_SHALLOW TRUE)
# v1.91.9b, not v1.91.0: the SDL3 platform backend in 1.91.0 predates SDL3's release and no longer
# compiles against SDL3 3.4 (SDL_SetRelativeMouseMode / SDL_CaptureMouse signatures). Measured on this
# box: 1.91.0 imgui_impl_sdl3.cpp = 2 errors, 1.91.9b = 0; imgui_impl_vulkan and the win32/dx12 pair
# compile clean on both, so the Windows build is unaffected.
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.9b GIT_SHALLOW TRUE)
# SDL3 for controllers, static so it links into the exe
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_RENDER OFF CACHE BOOL "" FORCE)
set(SDL_GPU OFF CACHE BOOL "" FORCE)
set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
FetchContent_Declare(SDL3 GIT_REPOSITORY https://github.com/libsdl-org/SDL.git GIT_TAG release-3.4.16 GIT_SHALLOW TRUE)
if(CMAKE_CROSSCOMPILING)
    # Zydis is only for the tools, which a cross build takes from the build machine (below)
    FetchContent_MakeAvailable(tomlplusplus imgui SDL3)
else()
    FetchContent_MakeAvailable(zydis tomlplusplus imgui SDL3)
endif()

if(CMAKE_CROSSCOMPILING)
# xbe_recomp runs during the build, so a cross build takes it from a build of cmake/XbeTools.cmake
# for this (the build) machine instead of building it for the target. xbe_symbols is only run by hand.
set(XBE_HOST_TOOLS_DIR "" CACHE PATH "build directory of cmake/XbeTools.cmake for the build machine (xbe_recomp)")
set(XBE_RECOMP_HOST ${XBE_HOST_TOOLS_DIR}/xbe_recomp${CMAKE_HOST_EXECUTABLE_SUFFIX})
if(NOT XBE_HOST_TOOLS_DIR OR NOT EXISTS ${XBE_RECOMP_HOST})
    message(FATAL_ERROR "cross-compiling runs xbe_recomp on this machine, and XBE_HOST_TOOLS_DIR=\"${XBE_HOST_TOOLS_DIR}\" "
        "holds no xbe_recomp${CMAKE_HOST_EXECUTABLE_SUFFIX}: build cmake/XbeTools.cmake for this machine first "
        "and point XBE_HOST_TOOLS_DIR at that build directory")
endif()
add_executable(xbe_recomp IMPORTED)
set_target_properties(xbe_recomp PROPERTIES IMPORTED_LOCATION ${XBE_RECOMP_HOST})
else()
# x86 to C++ recompiler
file(GLOB XBE_RECOMP_SOURCES CONFIGURE_DEPENDS ${XBE_SDK_DIR}/tools/recomp/*.cpp)
add_executable(xbe_recomp ${XBE_RECOMP_SOURCES})
target_include_directories(xbe_recomp PRIVATE ${XBE_SDK_DIR}/runtime ${XBE_SDK_DIR}/tools/recomp)
target_compile_definitions(xbe_recomp PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(xbe_recomp PRIVATE Zydis tomlplusplus::tomlplusplus)

# XDK signature scanner that writes a new title's config
add_executable(xbe_symbols ${XBE_SDK_DIR}/tools/symbols/main.cpp)
target_include_directories(xbe_symbols PRIVATE ${XBE_SDK_DIR}/runtime)
target_compile_definitions(xbe_symbols PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(xbe_symbols PRIVATE Zydis tomlplusplus::tomlplusplus)
endif()

if(WIN32)
set(XBE_IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_dx12.cpp)
else()
set(XBE_IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)
endif()

file(GLOB XBE_RUNTIME_SOURCES CONFIGURE_DEPENDS ${XBE_SDK_DIR}/runtime/*/*.cpp)
if(WIN32)
# the Linux app/GPU/audio counterparts stay out of the Windows link entirely
list(REMOVE_ITEM XBE_RUNTIME_SOURCES
    ${XBE_SDK_DIR}/runtime/app/main_sdl.cpp
    ${XBE_SDK_DIR}/runtime/app/image_decode.cpp
    ${XBE_SDK_DIR}/runtime/gpu/vulkan.cpp
    ${XBE_SDK_DIR}/runtime/gpu/spirv.cpp
    ${XBE_SDK_DIR}/runtime/audio/sdl_audio.cpp)
else()
# Win32-only sources out; the SDL/Vulkan counterparts in (globbed above,
# appended explicitly so the contract survives a stale glob)
list(REMOVE_ITEM XBE_RUNTIME_SOURCES
    ${XBE_SDK_DIR}/runtime/app/main.cpp
    ${XBE_SDK_DIR}/runtime/gpu/d3d12.cpp
    ${XBE_SDK_DIR}/runtime/audio/xaudio2.cpp)
list(APPEND XBE_RUNTIME_SOURCES
    ${XBE_SDK_DIR}/runtime/app/main_sdl.cpp
    ${XBE_SDK_DIR}/runtime/gpu/vulkan.cpp
    ${XBE_SDK_DIR}/runtime/audio/sdl_audio.cpp)
if(EXISTS ${XBE_SDK_DIR}/runtime/gpu/spirv.cpp)
    list(APPEND XBE_RUNTIME_SOURCES ${XBE_SDK_DIR}/runtime/gpu/spirv.cpp)
endif()
list(REMOVE_DUPLICATES XBE_RUNTIME_SOURCES)
find_package(Vulkan REQUIRED)
find_package(ZLIB REQUIRED)  # runtime/app/image_decode.cpp: the splash PNG decoder
# Shader compilers for the Vulkan backend. runtime/gpu/spirv.cpp carries both paths and picks at run
# time: DXC is dlopen'd (libdxcompiler.so), so it is never linked - it only needs XBE_HAVE_DXC and its
# headers; glslang is linked and enabled by its own headers being present (__has_include). Both are
# offered when found. XBE_SHADER_BACKEND makes one of them required instead of optional.
set(XBE_SHADER_BACKEND "" CACHE STRING "require a Vulkan shader compiler: dxc or glslang (empty: whatever is found)")
set_property(CACHE XBE_SHADER_BACKEND PROPERTY STRINGS dxc glslang)
find_path(XBE_DXC_INCLUDE dxc/dxcapi.h HINTS /opt/dxc/include)
find_library(XBE_GLSLANG glslang)
find_library(XBE_SPIRV SPIRV)
find_library(XBE_GLSLANG_RES glslang-default-resource-limits)
find_path(XBE_GLSLANG_INCLUDE glslang/Public/ShaderLang.h)
set(XBE_SHADER_LIBS "")
set(XBE_SHADER_DEFINES "")
set(XBE_SHADER_INCLUDES "")
if(XBE_DXC_INCLUDE)
    list(APPEND XBE_SHADER_DEFINES XBE_HAVE_DXC=1)
    list(APPEND XBE_SHADER_INCLUDES ${XBE_DXC_INCLUDE})
    set(XBE_HAVE_DXC_FOUND TRUE)
endif()
if(XBE_GLSLANG AND XBE_SPIRV AND XBE_GLSLANG_RES AND XBE_GLSLANG_INCLUDE)
    list(APPEND XBE_SHADER_LIBS ${XBE_GLSLANG} ${XBE_SPIRV} ${XBE_GLSLANG_RES})
    list(APPEND XBE_SHADER_INCLUDES ${XBE_GLSLANG_INCLUDE})
    set(XBE_HAVE_GLSLANG_FOUND TRUE)
endif()
if(XBE_SHADER_BACKEND STREQUAL "dxc" AND NOT XBE_HAVE_DXC_FOUND)
    message(FATAL_ERROR "XBE_SHADER_BACKEND=dxc but dxc/dxcapi.h was not found (expected /opt/dxc/include)")
elseif(XBE_SHADER_BACKEND STREQUAL "glslang" AND NOT XBE_HAVE_GLSLANG_FOUND)
    message(FATAL_ERROR "XBE_SHADER_BACKEND=glslang but glslang/SPIRV/glslang-default-resource-limits were not all found")
elseif(NOT XBE_HAVE_DXC_FOUND AND NOT XBE_HAVE_GLSLANG_FOUND)
    message(FATAL_ERROR "no HLSL to SPIR-V compiler: install DXC (headers in /opt/dxc/include) or glslang with its HLSL front end")
endif()
message(STATUS "Vulkan shader compilers: DXC=${XBE_HAVE_DXC_FOUND} glslang=${XBE_HAVE_GLSLANG_FOUND}")
endif()
# SDL's headers go to the input code alone (it is linked LINK_ONLY), so the generated sources' compile line stays the same
set_source_files_properties(${XBE_SDK_DIR}/runtime/core/input.cpp PROPERTIES INCLUDE_DIRECTORIES "${sdl3_SOURCE_DIR}/include")

# One executable per recompiled image: its code from config/<config> (partitions as that config sets them),
# the SDK runtime, XBE_HOST_SOURCES shared by every image of the game, and src/<game_dir> hooks only this image has.
# Expects assets/<xbe> and the config's out = "generated/<game_dir>". At run time the game may be a disc image instead.
function(add_xbe_image target xbe config partitions game_dir)
    set(GEN ${CMAKE_CURRENT_SOURCE_DIR}/generated/${game_dir})
    set(GEN_SOURCES ${GEN}/x86_func_mapping.cpp)
    math(EXPR last "${partitions} - 1")
    foreach(i RANGE ${last})
        list(APPEND GEN_SOURCES ${GEN}/x86_recomp.${i}.cpp)
    endforeach()
    file(GLOB configs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/config/*.toml ${XBE_SDK_DIR}/xdk/*/*.toml)
    add_custom_command(
        OUTPUT ${GEN}/codegen.stamp
        BYPRODUCTS ${GEN_SOURCES} ${GEN}/x86_recomp_shared.h ${GEN}/image_symbols.h
        COMMAND xbe_recomp config/${config} --xdk ${XBE_SDK_DIR}/xdk
        COMMAND ${CMAKE_COMMAND} -E touch ${GEN}/codegen.stamp
        DEPENDS xbe_recomp ${configs} ${CMAKE_CURRENT_SOURCE_DIR}/assets/${xbe}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Recompiling ${xbe}"
        VERBATIM)
    add_custom_target(recompile_${target} DEPENDS ${GEN}/codegen.stamp)
    set_source_files_properties(${GEN_SOURCES} PROPERTIES GENERATED TRUE
        COMPILE_OPTIONS "-O2;-fno-strict-aliasing;-Wno-unused-label;-Wno-parentheses-equality;-Wno-unused-variable")
    file(GLOB GAME_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${game_dir}/*.cpp)
    # the xbe's SHA-1: the executable checks the game it finds is the release it was made from
    set(xbe_file ${CMAKE_CURRENT_SOURCE_DIR}/assets/${xbe})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${xbe_file})
    file(SHA1 ${xbe_file} xbe_sha1)
    set(XBE_HASH ${CMAKE_CURRENT_BINARY_DIR}/${target}_xbe.cpp)
    file(CONFIGURE OUTPUT ${XBE_HASH} CONTENT "extern const char kImageSha1[] = \"${xbe_sha1}\";\n")
    if(WIN32)
    add_executable(${target} WIN32 ${XBE_RUNTIME_SOURCES} ${XBE_HOST_SOURCES} ${GAME_SOURCES} ${GEN_SOURCES} ${XBE_IMGUI_SOURCES} ${XBE_HASH})
    else()
    add_executable(${target} ${XBE_RUNTIME_SOURCES} ${XBE_HOST_SOURCES} ${GAME_SOURCES} ${GEN_SOURCES} ${XBE_IMGUI_SOURCES} ${XBE_HASH})
    endif()
    add_dependencies(${target} recompile_${target})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${XBE_SDK_DIR}/runtime ${GEN})
    target_include_directories(${target} SYSTEM PRIVATE ${imgui_SOURCE_DIR})
    target_compile_options(${target} PRIVATE -fno-strict-aliasing)
    target_compile_definitions(${target} PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    if(WIN32)
    target_link_libraries(${target} PRIVATE tomlplusplus::tomlplusplus $<LINK_ONLY:SDL3::SDL3-static> d3d12 dxgi dxguid d3dcompiler xaudio2 ole32 windowscodecs dbghelp winmm dwmapi ws2_32)
    else()
    # SDL3 plain, not LINK_ONLY: the app layer (main_sdl, vulkan surface) needs its headers too.
    # DXC is dlopen'd by gpu/spirv.cpp and so is never linked, only included.
    target_link_libraries(${target} PRIVATE tomlplusplus::tomlplusplus SDL3::SDL3-static Vulkan::Vulkan
        ${XBE_SHADER_LIBS} ZLIB::ZLIB pthread dl)
    target_compile_definitions(${target} PRIVATE ${XBE_SHADER_DEFINES})
    target_include_directories(${target} SYSTEM PRIVATE ${XBE_SHADER_INCLUDES})
    endif()
    # lld keeps the recompiled weak functions overridable by the host ones; the debug link writes the
    # PDB that names them in a Windows crash dump
    if(MSVC)
    # clang-cl: CMake runs CMAKE_LINKER itself rather than through the compiler driver
    if(NOT CMAKE_LINKER MATCHES "lld-link")
        message(FATAL_ERROR "the recompiled image links with lld-link; CMAKE_LINKER is ${CMAKE_LINKER}")
    endif()
    target_link_options(${target} PRIVATE /DEBUG)
    else()
    target_link_options(${target} PRIVATE -fuse-ld=lld -g)
    endif()
endfunction()
