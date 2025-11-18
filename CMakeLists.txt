cmake_minimum_required(VERSION 3.14)
project(mitmnns LANGUAGES CXX)

# Header-only library
add_library(mitmnns INTERFACE)

target_include_directories(mitmnns INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# -----------------------------
# Install headers
# -----------------------------
install(
    DIRECTORY include/
    DESTINATION include
)

# -----------------------------
# Export the target
# -----------------------------
# Build tree 用
export(
    TARGETS mitmnns
    NAMESPACE mitmnns::
    FILE mitmnnsTargets.cmake
)

# Install tree 用
install(
    TARGETS mitmnns
    EXPORT mitmnnsTargets
)

install(
    EXPORT mitmnnsTargets
    NAMESPACE mitmnns::
    DESTINATION lib/cmake/mitmnns
)

# -----------------------------
# Config + Version files
# -----------------------------
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mitmnnsConfigVersion.cmake"
    VERSION "1.0.0"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mitmnnsConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mitmnnsConfig.cmake"
    INSTALL_DESTINATION lib/cmake/mitmnns
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/mitmnnsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mitmnnsConfigVersion.cmake"
    DESTINATION lib/cmake/mitmnns
)

# -----------------------------
# uninstall target
# -----------------------------
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
