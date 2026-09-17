cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Headers and libmujoco come from the installed mujoco wheel.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MUJOCO_PKG_DIR
  COMMAND_ERROR_IS_FATAL ANY)
if(WIN32)
  set(MUJOCO_DLL "${MUJOCO_PKG_DIR}/mujoco.dll")
  if(NOT EXISTS "${MUJOCO_DLL}")
    message(FATAL_ERROR "Could not find mujoco.dll in ${MUJOCO_PKG_DIR}")
  endif()

  # The MuJoCo wheel exports the C API from mujoco.dll but does not ship an
  # MSVC import library. Generate one at build time; it is not installed.
  get_filename_component(MSVC_TOOL_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
  find_program(MSVC_LIB_EXECUTABLE NAMES lib HINTS "${MSVC_TOOL_BIN_DIR}")
  if(NOT MSVC_LIB_EXECUTABLE)
    message(FATAL_ERROR "Could not find MSVC lib.exe")
  endif()
  find_program(MSVC_DUMPBIN_EXECUTABLE NAMES dumpbin HINTS "${MSVC_TOOL_BIN_DIR}")
  if(NOT MSVC_DUMPBIN_EXECUTABLE)
    message(FATAL_ERROR "Could not find MSVC dumpbin.exe")
  endif()

  execute_process(
    COMMAND "${MSVC_DUMPBIN_EXECUTABLE}" "/EXPORTS" "${MUJOCO_DLL}"
    OUTPUT_VARIABLE MUJOCO_EXPORTS
    COMMAND_ERROR_IS_FATAL ANY)
  string(REGEX MATCHALL
         " +[0-9]+ +[0-9A-Fa-f]+ +[0-9A-Fa-f]+ +([^ \r\n]+)"
         MUJOCO_EXPORT_MATCHES "${MUJOCO_EXPORTS}")
  if(NOT MUJOCO_EXPORT_MATCHES)
    message(FATAL_ERROR "Could not parse exports from ${MUJOCO_DLL}")
  endif()

  set(MUJOCO_DEF_FILE "${CMAKE_CURRENT_BINARY_DIR}/mujoco.def")
  file(WRITE "${MUJOCO_DEF_FILE}" "LIBRARY mujoco.dll\nEXPORTS\n")
  foreach(export_match IN LISTS MUJOCO_EXPORT_MATCHES)
    string(REGEX REPLACE ".* +([^ \r\n]+)$" "\\1" export_symbol "${export_match}")
    file(APPEND "${MUJOCO_DEF_FILE}" "  ${export_symbol}\n")
  endforeach()

  set(MUJOCO_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/mujoco.lib")
  execute_process(
    COMMAND "${MSVC_LIB_EXECUTABLE}" "/def:${MUJOCO_DEF_FILE}" "/machine:x64"
            "/out:${MUJOCO_LIBRARY}"
    COMMAND_ERROR_IS_FATAL ANY)
else()
  file(GLOB MUJOCO_LIBRARY "${MUJOCO_PKG_DIR}/libmujoco*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
  if(NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR "Could not find libmujoco in ${MUJOCO_PKG_DIR}")
  endif()
  list(GET MUJOCO_LIBRARY 0 MUJOCO_LIBRARY)
endif()

nanobind_add_module(_bindings FREE_THREADED NOMINSIZE src/mjbatch/csrc/bindings.cpp)
if(WIN32)
  # Windows headers are nested one directory deeper than the Unix wheels.
  target_include_directories(_bindings PRIVATE "${MUJOCO_PKG_DIR}/include/mujoco")
else()
  target_include_directories(_bindings PRIVATE "${MUJOCO_PKG_DIR}/include")
endif()
target_link_libraries(_bindings PRIVATE "${MUJOCO_LIBRARY}")

# Find libmujoco next to the installed extension, <site-packages>/mujoco. The
# macOS wheel's install name is a framework path that does not exist, so point
# our load command at the plain file name instead.
if(APPLE)
  get_filename_component(MUJOCO_LIB_NAME "${MUJOCO_LIBRARY}" NAME)
  execute_process(COMMAND otool -D "${MUJOCO_LIBRARY}" OUTPUT_VARIABLE _otool)
  string(REGEX MATCH "@rpath[^\n]*" MUJOCO_INSTALL_NAME "${_otool}")
  add_custom_command(TARGET _bindings POST_BUILD COMMAND install_name_tool -change
    "${MUJOCO_INSTALL_NAME}" "@rpath/${MUJOCO_LIB_NAME}" $<TARGET_FILE:_bindings>)
  set_target_properties(_bindings PROPERTIES INSTALL_RPATH "@loader_path/../mujoco")
elseif(NOT WIN32)
  set_target_properties(_bindings PROPERTIES INSTALL_RPATH "$ORIGIN/../mujoco")
endif()

install(TARGETS _bindings DESTINATION mjbatch)
