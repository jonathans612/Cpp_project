# **GPU-Accelerated N-Body Simulation in C++ and CUDA**

*A brief, high-quality GIF showing the simulation running with a high particle count. Crucially, it should demonstrate the dramatic performance difference when toggling between the CPU and GPU modes.*

## **Synopsis**

This project is a real-time N-body gravity simulation built from scratch in C++ and accelerated with NVIDIA CUDA. It simulates the gravitational interactions between thousands of particles, visualizing their movement and clustering in 2D. The core of the project is a direct comparison between a classic, single-threaded CPU implementation and a massively parallel GPU implementation, highlighting the performance benefits of High-Performance Computing (HPC) for scientific problems.

## **Key Features**

* **Real-time N-Body Simulation:** Simulates gravitational forces using Newton's Law of Universal Gravitation.  
* **Stable Physics Engine:** Implements a Kick-Drift-Kick Leapfrog integrator for energy conservation and stable orbits.  
* **Plummer Softening:** Avoids numerical instability from division-by-zero errors when particles are too close.  
* **GPU Acceleration:** Offloads the computationally intensive O(N²) force calculation to the GPU using a custom CUDA kernel.  
* **Live Performance Toggle:** Switch between the CPU and GPU calculation methods at runtime to instantly see the performance difference.  
* **Configurable Build System:** Uses CMake to manage dependencies and allows for easy switching between CPU-only and CUDA-enabled builds.

## **Skills Demonstrated**

* **Programming Languages:** C++, CUDA C++  
* **High-Performance Computing (HPC):** Parallel programming with CUDA, identifying and optimizing computational bottlenecks.  
* **Scientific Computing:** Implementation of numerical methods (Leapfrog integrator) and physics models (N-body problem).  
* **C++ Development:** Object-oriented design (Particle struct/class), memory management, and use of the Standard Library (std::vector).  
* **Build Systems:** Cross-platform project configuration with CMake, including linking external libraries and managing CUDA projects.  
* **Graphics:** Basic real-time rendering and window/input management using OpenGL and GLFW.

## **Performance Comparison**

The primary bottleneck in an N-body simulation is the O(N²) force calculation. Offloading this to the GPU provides a dramatic speedup. The following benchmarks were recorded by disabling VSync to measure raw uncapped framerates.

| Number of Particles | CPU Framerate (FPS) | GPU Framerate (FPS) | Speedup Factor |  
| 500 | \~77 FPS | \~2100 FPS | \~27x |  
| 800 | \~30 FPS | \~1800 FPS | \~60x |  
| 1,000 | \~20 FPS | \~1600 FPS | \~80x |  
| 5,000 | \~1 FPS (unusable) | \~500 FPS | \~500x |

## **Building and Running the Project**

### **Prerequisites**

1. **NVIDIA CUDA Toolkit:** Version 11.0 or newer. ([Download here](https://developer.nvidia.com/cuda-downloads))  
2. **C++ Compiler:** A modern C++ compiler (MSVC on Windows, GCC/Clang on Linux).  
3. **CMake:** Version 3.10 or newer. ([Download here](https://cmake.org/download/))  
4. An NVIDIA GPU with CUDA support.

### **Steps to Build**

1. **Clone the repository:**  
   git clone [https://github.com/jonathans612/Cpp\_project.git](https://github.com/jonathans612/Cpp_project.git)  
   cd your-repo-name

2. **Create a build directory:**  
   mkdir build && cd build

3. **Run CMake:**  
   * **For the CUDA (GPU) version:**  
     cmake ..

   * **For the CPU-only version:** (Turn the USE\_CUDA option OFF)  
     cmake .. \-DUSE\_CUDA=OFF

4. **Compile the project:**  
   * **On Windows (Visual Studio):** Open the .sln file in the build directory and click "Build Solution".  
   * **On Linux (Make):** Run make from the build directory.

### **Running the Executable**

The executable (nbody\_sim.exe or nbody\_sim) will be located in the build/Debug (or build) directory. After building, you can run it from your terminal or by double-clicking it.

## **Controls**

* **G Key:** Switch to **GPU** force calculation (fast).  
* **C Key:** Switch to **CPU** force calculation (slow).  
* **ESC Key:** Close the application.

## **Technical Details & Concepts Learned**

This project was a practical exercise in applying computer science fundamentals to a scientific problem.

* **The O(N²) Bottleneck:** The core challenge was identifying that the nested loop for force calculation scaled quadratically with the number of particles, making the CPU version unusable for large N.  
* **Parallelization with CUDA:** The force calculation for each particle is an independent task ("embarrassingly parallel"). This makes it a perfect candidate for the thousands of cores on a modern GPU. A custom CUDA kernel was written where each thread is responsible for calculating the net force on a single particle.  
* **Leapfrog Integration:** To ensure the simulation was physically stable over time, a Kick-Drift-Kick Leapfrog integrator was used instead of the simpler but less stable Euler method. This conserves energy in the system, leading to realistic and non-decaying orbits.  
* **CMake for Hybrid Projects:** CMake was essential for managing a project that could be built in two different ways (CPU-only or C++/CUDA). The use of option() and conditional logic allows for a clean separation of build concerns.

## **Future Work**

* **Modernize Rendering:** Replace the legacy "immediate mode" OpenGL with modern OpenGL using Vertex Buffer Objects (VBOs) and shaders for a massive rendering performance boost.  
* **Implement Adaptive Timestepping:** Move from a global timestep to an adaptive one, where particles in dense, fast-moving regions are updated more frequently than isolated particles.  
* **Optimize CUDA Kernel:** Explore more advanced CUDA optimizations, such as using shared memory to reduce global memory reads.