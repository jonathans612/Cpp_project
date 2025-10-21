cmake_minimum_required(VERSION 3.10)

# --- Project Setup and Options ---

# Define an option to toggle CUDA on or off (default is ON)
option(USE_CUDA "Enable GPU acceleration with CUDA" ON)

# Set the project name and languages based on the option
if(USE_CUDA)
    project(NBodySimulation_cuda LANGUAGES CXX CUDA)
else()
    project(NBodySimulation LANGUAGES CXX)
endif()

# Set the C++ standard for the project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Dependencies ---
find_package(OpenGL REQUIRED)

# Manually find GLEW and GLFW (common to both builds)
find_path(GLEW_INCLUDE_DIR NAMES GL/glew.h PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libs/glew-2.1.0/include")
find_library(GLEW_LIBRARY NAMES glew32.lib PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libs/glew-2.1.0/lib/Release/x64")
find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw-3.4.bin.WIN64/include")
find_library(GLFW_LIBRARY NAMES glfw3.lib PATHS "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw-3.4.bin.WIN64/lib-vc2022")

# --- Create Executable and Find Specific Dependencies ---
if(USE_CUDA)
    # --- CUDA Build Configuration ---
    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP COMPONENTS CXX REQUIRED) # Find OpenMP just for C++
    set(CMAKE_CUDA_ARCHITECTURES "native")
    add_executable(nbody_sim main.cu)
else()
    # --- CPU-Only Build Configuration ---
    find_package(OpenMP REQUIRED) # Find OpenMP (no CUDA to worry about)
    add_executable(nbody_sim main_cpu.cpp)
endif()

# --- Add Post-Build Command to Copy DLLs (common to both) ---

# This command runs after the nbody_sim target is built
add_custom_command(
    TARGET nbody_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/glew-2.1.0/bin/Release/x64/glew32.dll"
        $<TARGET_FILE_DIR:nbody_sim>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw-3.4.bin.WIN64/lib-vc2022/glfw3.dll"
        $<TARGET_FILE_DIR:nbody_sim>
    COMMENT "Copying required DLLs to output directory"
)

# --- Link Libraries ---

# Link common include directories to our executable
target_include_directories(nbody_sim PRIVATE 
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
)

# Link common libraries to our executable
target_link_libraries(nbody_sim PRIVATE 
    ${OPENGL_LIBRARY}
    ${GLEW_LIBRARY}
    ${GLFW_LIBRARY}
)

# Conditionally link CUDA and OpenMP
if(USE_CUDA)
    # For the CUDA build, link both CUDA and OpenMP
    target_link_libraries(nbody_sim PRIVATE CUDA::cudart OpenMP::OpenMP_CXX)
else()
    # For the CPU build, link only OpenMP
    target_link_libraries(nbody_sim PRIVATE OpenMP::OpenMP_CXX)
endif()