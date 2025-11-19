#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>   // Required for std::unique_ptr (Smart Pointers)
#include <random>   // Required for modern C++ RNG (std::mt19937)
#include <functional> 
#include <algorithm> // Required for std::fill
#include <string>    // Required for std::stoi

// --- Constants ---
// usage of 'constexpr' allows the compiler to inline these values for performance
constexpr int WINDOW_WIDTH = 800;
constexpr int WINDOW_HEIGHT = 600;
constexpr float G_CONST = 1.0f;   // Gravitational Constant (Tuned for visual effect)
constexpr float EPSILON = 10.0f;  // Softening parameter to prevent division by zero at r=0
constexpr float DT = 1.0f;        // Time step size

// Compile-time toggle: Set to true for deterministic debugging, false for random runs.
constexpr bool USE_FIXED_SEED = true; 

// --- CUDA Kernel ---
// 'calculateForcesKernel': Computes gravitational forces for N particles.
// 
// OPTIMIZATION NOTE: usage of '__restrict__' keyword.
// Allows the compiler to cache reads more aggressively, improving bandwidth usage.
__global__ void calculateForcesKernel(const float* __restrict__ posX, 
                                      const float* __restrict__ posY, 
                                      const float* __restrict__ mass, 
                                      float* __restrict__ forceX, 
                                      float* __restrict__ forceY, 
                                      int N, float G, float eps) 
{
    // Global Thread ID calculation
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: Prevent threads from accessing memory outside particle count
    if (i >= N) return;

    // Local Register Caching:
    // Load "my" data into registers once to avoid repeated global memory lookups.
    float myX = posX[i];
    float myY = posY[i];
    float myMass = mass[i];
    float fx = 0.0f;
    float fy = 0.0f;

    // The N^2 loop: Compute interaction with every other particle
    for (int j = 0; j < N; ++j) {
        if (i == j) continue; // Skip self-interaction

        float dx = posX[j] - myX;
        float dy = posY[j] - myY;
        float rSq = dx*dx + dy*dy;
        
        // PHYSICS NOTE: Plummer Softening
        // We add eps^2 to distance^2 to avoid singularities when particles collide.
        float distSqr = rSq + eps*eps;
        
        // OPTIMIZATION NOTE: rsqrtf()
        // We need 1 / dist^3 for the force vector calculation.
        // Instead of pow(dist, -1.5), GPU hardware intrinsic rsqrtf (1/sqrt).
        // Sgnificantly faster than standard math functions on CUDA cores.
        float invDistCube = rsqrtf(distSqr * distSqr * distSqr); 

        float f = G * myMass * mass[j] * invDistCube;
        
        fx += f * dx;
        fy += f * dy;
    }

    // Write result back to Global Memory
    forceX[i] = fx;
    forceY[i] = fy;
}

// --- Helper: Modern CUDA Error Checker ---
// Throws standard C++ exceptions instead of exiting the program abruptly.
// This allows the application to handle errors (e.g., fallback to CPU) if desired.
void checkCudaStatus(cudaError_t err, const std::string& msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(msg + ": " + cudaGetErrorString(err));
    }
}

// --- CUDA Smart Pointer Helper (RAII) ---
// Custom deleter functor: Defines HOW to free the memory when the unique_ptr dies.
struct CudaDeleter {
    void operator()(float* ptr) const {
        cudaFree(ptr);
    }
};

// Type alias for a unique_ptr that holds a float* and uses our Custom Deleter.
using CudaPtr = std::unique_ptr<float, CudaDeleter>;

// Factory function to allocate GPU memory and wrap it in RAII safety immediately.
CudaPtr allocateDevice(size_t size) {
    float* raw_ptr = nullptr;
    checkCudaStatus(cudaMalloc(&raw_ptr, size), "CUDA Malloc Failed");
    return CudaPtr(raw_ptr); // Transfer ownership to unique_ptr
}

// --- Simulation Class ---
// Encapsulates all simulation state, resources, and logic.
// Follows the RAII (Resource Acquisition Is Initialization) pattern:
// - Constructor: Acquires resources (Window, Memory, GPU).
// - Destructor: Releases resources (Window, Memory via smart pointers).
class NBodySimulation {
private:
    int numParticles;
    
    // Host (CPU) Data: std::vector manages heap memory automatically.
    // Structure of Arrays (SoA) layout used here (separate X/Y/Mass arrays) 
    // often allows better SIMD/Coalescing than Array of Structures (AoS).
    std::vector<float> h_posX, h_posY, h_mass;
    std::vector<float> h_velX, h_velY;
    std::vector<float> h_forceX, h_forceY;

    // Device (GPU) Data: Smart pointers manage GPU heap memory.
    CudaPtr d_posX, d_posY, d_mass, d_forceX, d_forceY;

    GLFWwindow* window = nullptr;
    bool useGpu = true;

    // FPS & Timing
    double lastTime = 0.0;
    int frameCount = 0;

    // VSync State
    bool vsyncEnabled = false; 
    bool lastVKeyState = false; // Debounce helper

public:
    NBodySimulation(int n) 
        : numParticles(n), 
          // Initialize smart pointers to nullptr
          d_posX(nullptr), d_posY(nullptr), 
          d_mass(nullptr), d_forceX(nullptr), d_forceY(nullptr)
    {
        initWindow();
        initParticles();
        initGPU(); // Note: If this throws, ~NBodySimulation is NOT called, 
                   // but members (vectors/unique_ptrs) are still destroyed safely.
    }

    ~NBodySimulation() {
        if (window) glfwDestroyWindow(window);
        glfwTerminate();
        // GPU memory is freed automatically by CudaPtr destructors here.
    }

    // Main Loop
    void run() {
        lastTime = glfwGetTime(); 

        while (!glfwWindowShouldClose(window)) {
            handleInput();
            updatePhysics();
            render();

            // --- FPS Logic ---
            double currentTime = glfwGetTime();
            frameCount++;

            if (currentTime - lastTime >= 1.0) {
                char title[256];
                snprintf(title, sizeof(title), "N-Body (%s) | VSync: %s | %d FPS", 
                         useGpu ? "GPU" : "CPU", 
                         vsyncEnabled ? "ON" : "OFF",
                         frameCount);
                glfwSetWindowTitle(window, title);
                frameCount = 0;
                lastTime = currentTime;
            }
        }
    }

private:
    void initWindow() {
        if (!glfwInit()) throw std::runtime_error("Failed to init GLFW");
        window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "N-Body (GPU Mode)", nullptr, nullptr);
        if (!window) throw std::runtime_error("Failed to create window");
        glfwMakeContextCurrent(window);
        glewInit();
        
        // Disable VSync initially for benchmarking raw performance
        glfwSwapInterval(0);

        // Set up 2D Orthographic Projection
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1);
        glMatrixMode(GL_MODELVIEW);
    }

    void initParticles() {
        // Resize CPU vectors
        h_posX.resize(numParticles); h_posY.resize(numParticles);
        h_mass.resize(numParticles);
        h_velX.resize(numParticles, 0.0f); h_velY.resize(numParticles, 0.0f);
        h_forceX.resize(numParticles); h_forceY.resize(numParticles);

        // Modern C++ Random Number Generation (Mersenne Twister)
        // Much higher quality randomness than rand()
        std::random_device rd;
        unsigned int seed = USE_FIXED_SEED ? 42 : rd();
        std::mt19937 gen(seed);
        
        std::uniform_real_distribution<float> posDistX(0, (float)WINDOW_WIDTH);
        std::uniform_real_distribution<float> posDistY(0, (float)WINDOW_HEIGHT);
        std::uniform_real_distribution<float> massDist(1.0f, 10.0f);

        // Initialize positions and masses
        for(int i=0; i<numParticles; ++i) {
            h_posX[i] = posDistX(gen);
            h_posY[i] = posDistY(gen);
            h_mass[i] = massDist(gen);
        }
    }

    void initGPU() {
        size_t bytes = numParticles * sizeof(float);
        
        // Allocation via helper
        d_posX = allocateDevice(bytes);
        d_posY = allocateDevice(bytes);
        d_mass = allocateDevice(bytes);
        d_forceX = allocateDevice(bytes);
        d_forceY = allocateDevice(bytes);

        // Initial data transfer: CPU -> GPU
        checkCudaStatus(cudaMemcpy(d_posX.get(), h_posX.data(), bytes, cudaMemcpyHostToDevice), "Memcpy Init");
        checkCudaStatus(cudaMemcpy(d_posY.get(), h_posY.data(), bytes, cudaMemcpyHostToDevice), "Memcpy Init");
        checkCudaStatus(cudaMemcpy(d_mass.get(), h_mass.data(), bytes, cudaMemcpyHostToDevice), "Memcpy Init");
    }

    // Core Physics Loop: Uses Leapfrog (Kick-Drift-Kick) Integration
    void updatePhysics() {
        
        // 1. First Kick (CPU): Update velocity by 1/2 time step
        for (int i = 0; i < numParticles; ++i) {
            h_velX[i] += 0.5f * (h_forceX[i] / h_mass[i]) * DT;
            h_velY[i] += 0.5f * (h_forceY[i] / h_mass[i]) * DT;
        }

        // 2. Drift (CPU): Update position by full time step
        for (int i = 0; i < numParticles; ++i) {
            h_posX[i] += h_velX[i] * DT;
            h_posY[i] += h_velY[i] * DT;
            
            // Simple boundary bounce
            if(h_posX[i] < 0 || h_posX[i] > WINDOW_WIDTH) h_velX[i] *= -1;
            if(h_posY[i] < 0 || h_posY[i] > WINDOW_HEIGHT) h_velY[i] *= -1;
        }

        // 3. Force Calculation (The Heavy Computation)
        if (useGpu) {
            size_t bytes = numParticles * sizeof(float);
            
            // Transfer updated positions to GPU
            // .get() retrieves the raw pointer from the smart pointer for the C-API
            checkCudaStatus(cudaMemcpy(d_posX.get(), h_posX.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H2D");
            checkCudaStatus(cudaMemcpy(d_posY.get(), h_posY.data(), bytes, cudaMemcpyHostToDevice), "Memcpy H2D");

            // Calculate Grid Dimensions
            int threads = 256;
            int blocks = (numParticles + threads - 1) / threads;
            
            // Launch Kernel
            calculateForcesKernel<<<blocks, threads>>>(
                d_posX.get(), d_posY.get(), d_mass.get(), 
                d_forceX.get(), d_forceY.get(), 
                numParticles, G_CONST, EPSILON
            );

            // Error checking for Kernel Launch (async errors) and Execution (sync errors)
            checkCudaStatus(cudaPeekAtLastError(), "Kernel Launch Failed");
            checkCudaStatus(cudaDeviceSynchronize(), "Kernel Execution Failed");

            // Retrieve forces from GPU
            checkCudaStatus(cudaMemcpy(h_forceX.data(), d_forceX.get(), bytes, cudaMemcpyDeviceToHost), "Memcpy D2H");
            checkCudaStatus(cudaMemcpy(h_forceY.data(), d_forceY.get(), bytes, cudaMemcpyDeviceToHost), "Memcpy D2H");
        } 
        else {
            // --- CPU FALLBACK (OpenMP) ---
            std::fill(h_forceX.begin(), h_forceX.end(), 0.0f);
            std::fill(h_forceY.begin(), h_forceY.end(), 0.0f);

            // #pragma omp parallel for: Uses multi-threading to speed up the outer loop
            #pragma omp parallel for
            for (int i = 0; i < numParticles; ++i) {
                for (int j = 0; j < numParticles; ++j) {
                    if (i == j) continue; 

                    float dx = h_posX[j] - h_posX[i];
                    float dy = h_posY[j] - h_posY[i];
                    float rSq = dx * dx + dy * dy;
                    
                    float denom = std::pow(rSq + EPSILON * EPSILON, 1.5f);
                    
                    if (denom > 0.0f) {
                        float force_scalar = (G_CONST * h_mass[i] * h_mass[j]) / denom;
                        // Atomic operations not needed here as each thread writes to a unique 'i'
                        h_forceX[i] += force_scalar * dx;
                        h_forceY[i] += force_scalar * dy;
                    }
                }
            }
        }

        // 4. Second Kick (CPU): Update velocity by remaining 1/2 time step
        for (int i = 0; i < numParticles; ++i) {
            h_velX[i] += 0.5f * (h_forceX[i] / h_mass[i]) * DT;
            h_velY[i] += 0.5f * (h_forceY[i] / h_mass[i]) * DT;
        }
    }

    void handleInput() {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) 
            glfwSetWindowShouldClose(window, true);
            
        // Mode Switching
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && useGpu) useGpu = false;
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS && !useGpu) useGpu = true;

        // VSync Toggle (Debounced)
        bool currentVState = (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS);
        if (currentVState && !lastVKeyState) {
            vsyncEnabled = !vsyncEnabled;
            glfwSwapInterval(vsyncEnabled ? 1 : 0);
            std::cout << "VSync Toggled: " << (vsyncEnabled ? "ON" : "OFF") << std::endl;
        }
        lastVKeyState = currentVState;
    }

    // Render Pipeline
    // Note: Uses glBegin/glEnd (Immediate Mode) for simplicity.
    // In a production engine, we would use VBOs (Vertex Buffer Objects) mapped to CUDA pointers.
    void render() {
        glClear(GL_COLOR_BUFFER_BIT);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        glColor3f(1.0f, 1.0f, 1.0f);
        for (int i = 0; i < numParticles; ++i) {
            glVertex2f(h_posX[i], h_posY[i]);
        }
        glEnd();
        glfwSwapBuffers(window);
    }
};

// Entry Point: Handles CLI arguments and Exception catching
int main(int argc, char* argv[]) {
    try {
        int particleCount = 800;

        // Simple CLI argument parsing
        if (argc > 1) {
            particleCount = std::stoi(argv[1]);
        }

        std::cout << "Initializing N-Body Simulation with " << particleCount << " particles..." << std::endl;
        std::cout << "Controls: [G] GPU | [C] CPU | [V] VSync" << std::endl;

        // Stack allocation: 'sim' is destroyed automatically when main exits.
        NBodySimulation sim(particleCount); 
        sim.run();
    } 
    catch (const std::exception& e) {
        // Catch any runtime_errors thrown during init or CUDA calls
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}