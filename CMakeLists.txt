# Project specific options :
#   - BP_BUILD_TESTS (requires BUILD_TESTING set to ON)
# Other options might be available through the cmake scripts including (not exhaustive):
#   - ENABLE_WARNINGS_SETTINGS
#   - ENABLE_LTO
#
cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source.")
endif()

# Put the project early since modules might need to detect the compiler.
# More information https://cmake.org/cmake/help/latest/command/project.html
project(
    "HDC" # This will exposed as the variable PROJECT_NAME.
    VERSION 0.1.0 # Used for installation and defines variables PROJECT_VERSION_MAJOR, PROJECT_VERSION_MINOR, PROJECT_VERSION_PATCH, and PROJECT_VERSION_TWEAK.
    LANGUAGES C CXX # Used to determine the languages to use based on file extensions
)

############################
##  Modules and scripts   ##
############################

# Standard CMake modules

include(CTest)                          # Must be called before adding tests but after calling project(). This automatically calls enable_testing() and configures ctest targets when using Make/Ninja
include(CMakeDependentOption)           # This is a really useful scripts that creates options that depends on other options. It can even be used with generator expressions !
include(CMakePackageConfigHelpers)      # Helper to create relocatable packages

# Custom modules and scripts

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake") # Make our cmake scripts available

include(LTO)
include(Warnings)
include(CopyDllsForDebug)
include(Coverage)

###############
##  OPTIONS  ##
###############

# You should try to give as much control over the project setup to the user.
# When modifying compile flags for example, if they are not mandatory, provide an option.

option(${PROJECT_NAME}_RUN_TESTS "Run Tests After Building" OFF)

# Use your own option for tests, in case people use your library through add_subdirectory
cmake_dependent_option(${PROJECT_NAME}_BUILD_TESTS
    "Enable ${PROJECT_NAME} project tests targets" ON # By default we want tests if CTest is enabled
    "BUILD_TESTING" OFF # Stay coherent with CTest variables
)

# External dependencies
add_subdirectory(external)

# It is always easier to navigate in an IDE when projects are organized in folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Whe building a shared library, you do not want to export all symbols by default
# gcc (and hence clang) are wrong about this.
#
# For more information, see https://gcc.gnu.org/wiki/Visibility and https://www.youtube.com/embed/m0DwB4OvDXk
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

###############
##  Project  ##
###############

# Check for LTO support (needs to be after project(...) )
find_lto(CXX)

find_package(PkgConfig REQUIRED)

find_package(raspicam REQUIRED)
find_package(OpenCV REQUIRED)

#==========================#
#  Main executable  #
#==========================#

# Always list the source files explicitly, including headers so that they are listed in the IDE
# If you need to use files based on a variable value, use target_sources
add_executable(HDC
    src/main.cpp
)

target_include_directories(HDC PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    external/bm-7.10.3/src
    include/
)

target_link_libraries(HDC PRIVATE
    fmt::fmt
    spdlog::spdlog
    dlib::dlib
)

# Require c++17, this is better than setting CMAKE_CXX_STANDARD since it won't pollute other targets
target_compile_features(HDC PRIVATE cxx_std_17)

# CMake scripts extensions
target_set_warnings(HDC ENABLE ALL DISABLE Annoying) # Helper that can set default warning flags for you
target_enable_lto(HDC optimized) #enable lto if available for non-debug configurations
copy_dlls_for_debug(HDC "" "") # Copy dependencies next to the executable (DLLs for example)

# Setup our project as the startup project for Visual so that people don't need to do it manually
set_directory_properties(PROPERTIES VS_STARTUP_PROJECT HDC)

#==========================#
#  AntiTheft executable  #
#==========================#

# Always list the source files explicitly, including headers so that they are listed in the IDE
# If you need to use files based on a variable value, use target_sources
add_executable(AntiTheft
    src/AntiTheft.cpp
)

target_include_directories(AntiTheft PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    external/bm-7.10.3/src
    include/
)

target_link_libraries(AntiTheft PRIVATE
    fmt::fmt
    spdlog::spdlog
    dlib::dlib
    ${raspicam_CV_LIBS}
    wiringPi
)

# Require c++17, this is better than setting CMAKE_CXX_STANDARD since it won't pollute other targets
target_compile_features(AntiTheft PRIVATE cxx_std_17)

# CMake scripts extensions
target_set_warnings(AntiTheft ENABLE ALL DISABLE Annoying) # Helper that can set default warning flags for you
target_enable_lto(AntiTheft optimized) #enable lto if available for non-debug configurations
copy_dlls_for_debug(AntiTheft "" "") # Copy dependencies next to the executable (DLLs for example)
