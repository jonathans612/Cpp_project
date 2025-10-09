cmake_minimum_required(VERSION 3.10)
project(MyHPCProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set paths for dependencies in the "libs" folder
set(CMAKE_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/include")
# Note: For GLEW, you might need to adjust this path depending on its internal structure
set(CMAKE_LIBRARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/lib/Release/x64")

# --- Find Dependencies ---

# Find OpenGL
find_package(OpenGL REQUIRED)

# Find GLEW (this should still work as before)
find_package(GLEW REQUIRED)

# Manually include the GLFW header directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw-3.4.bin.WIN64/include")

# Find the GLFW library file and save its path to the GLFW_LIBRARY variable
find_library(GLFW_LIBRARY
    NAMES glfw3
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw-3.4.bin.WIN64/lib-vc2022"
)

# --- Build Executable ---

# Add executable
add_executable(my_app main.cpp)

# Link all dependencies to the executable
target_link_libraries(my_app PRIVATE
    OpenGL::GL
    GLEW::glew
    ${GLFW_LIBRARY}
)