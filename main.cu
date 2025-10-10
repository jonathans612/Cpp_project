#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath> // For sqrt and pow
#include <string> // For std::string in error checker

// CUDA error checking function
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Macro that wraps the function call
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// --- Simulation Constants ---
const bool USE_FIXED_SEED = true; // Set to 'true' for reproducible, 'false' for random
const float G = 1.0f;             // 6.674e-5 // A "tuned" gravitational constant for our simulation
const float epsilon = 10.0f;
const int numParticles = 5000;    // Number of particles in the simulation
const float dt = 1.0f;            // Our time step
const int MAX_PARTICLES = 10000;  // A reasonable cap, adjust for your hardware

// --- CUDA Kernel ---
__global__ void calculateForces(float* posX, float* posY, float* mass, 
                            float* forceX, float* forceY, 
                            int N, float G, float epsilon) 
{
    // 1. Calculate the unique global ID for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Boundary Check: Ensure this thread is within the bounds of our particle array
    if (idx < N) {
        // 3. Load this thread's particle data into local variables
        float myPosX = posX[idx]; // Position
        float myPosY = posY[idx];
        float myMass = mass[idx];
        
        // 4. Initialize force accumulators for this thread
        float fx = 0.0f;
        float fy = 0.0f;

        // 5. Loop through all other particles to calculate forces
        for (int j = 0; j < N; ++j) {
            if (idx == j) continue; // Don't calculate force with self

            // Load the other particle's data
            float otherPosX = posX[j];
            float otherPosY = posY[j];
            float otherMass = mass[j];

            // Perform the same Plummer force calculation as the CPU version
            float dx = otherPosX - myPosX;
            float dy = otherPosY - myPosY;
            float rSq = dx * dx + dy * dy;
            float denom = powf(rSq + epsilon * epsilon, 1.5f);
            
            // Avoid division by zero if denom is somehow zero
            if (denom > 0) {
                float force_scalar = (G * myMass * otherMass) / denom;
                fx += force_scalar * dx;
                fy += force_scalar * dy;
            }
        }

        // 6. Write the final calculated force back to global memory
        forceX[idx] = fx;
        forceY[idx] = fy;
    }
}

int main(void) {
    if (numParticles > MAX_PARTICLES) {
        std::cerr << "Error: Particle count " << numParticles 
                << " exceeds the maximum of " << MAX_PARTICLES 
                << ". Please reduce the particle count." << std::endl;
        return -1;
    }
    
    GLFWwindow* window;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "My HPC Project", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Force the CUDA context to initialize
    gpuErrchk(cudaFree(0));

    // --- Set up the coordinate system ---
    int width, height;
    glfwGetFramebufferSize(window, &width, &height); // Get window dimensions
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, width, 0.0, height, -1.0, 1.0); // Set coordinate system to match pixels
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Set the background color
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

    // --- Create Particles ---
    if (USE_FIXED_SEED) {
        srand(42); // Use a fixed seed
    } else {
        srand(time(0)); // Use a random seed
    }

    // ... create vectors ...
    std::vector<float> h_posX, h_posY, h_velX, h_velY, h_forceX, h_forceY, h_mass;
    size_t dataSize = numParticles * sizeof(float);
    
    try {
        h_posX.resize(numParticles);
        h_posY.resize(numParticles);
        h_velX.assign(numParticles, 0.0f);
        h_velY.assign(numParticles, 0.0f);
        h_forceX.assign(numParticles, 0.0f);
        h_forceY.assign(numParticles, 0.0f);
        h_mass.resize(numParticles);
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Error: Not enough host (CPU) memory to allocate for "
                << numParticles << " particles. " << e.what() << std::endl;
        return -1;
    }

    for (int i = 0; i < numParticles; ++i) {
        h_posX[i] = rand() % width;
        h_posY[i] = rand() % height;
        h_mass[i] = (rand() % 1000 / 100.0f) + 1.0f;  // Random mass between 1.0 and 11.0
    }

    // --- Proactive Memory Check ---
    size_t freeMem, totalMem;
    gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));

    // Calculate memory needed for our 5 float arrays (posX, posY, mass, forceX, forceY)
    size_t requiredMem = numParticles * sizeof(float) * 5;

    std::cout << "Required GPU Memory: " << requiredMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Free GPU Memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;

    if (requiredMem > freeMem) {
        std::cerr << "Error: Not enough free GPU memory to allocate for " 
                << numParticles << " particles." << std::endl;
        return -1; // Exit gracefully
    }
    
    // --- Device (GPU) Memory Allocation ---
    float *d_posX, *d_posY, *d_forceX, *d_forceY, *d_mass;

    gpuErrchk(cudaMalloc(&d_posX, dataSize));
    gpuErrchk(cudaMalloc(&d_posY, dataSize));
    gpuErrchk(cudaMalloc(&d_mass, dataSize));
    gpuErrchk(cudaMalloc(&d_forceX, dataSize));
    gpuErrchk(cudaMalloc(&d_forceY, dataSize));

    gpuErrchk(cudaMemcpy(d_mass, h_mass.data(), dataSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_posX, h_posX.data(), dataSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_posY, h_posY.data(), dataSize, cudaMemcpyHostToDevice));

    // Toggle for switching between CPU and GPU
    bool use_gpu = true;
    
    // Main loop: runs until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // --- INPUT HANDLING ---
        // Check if the ESC key is pressed and set the window to close
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        // Check for 'G' key to switch to GPU
        if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
            use_gpu = true;
            glfwSetWindowTitle(window, "N-Body Simulation (GPU)");
        }

        // Check for 'C' key to switch to CPU
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
            use_gpu = false;
            glfwSetWindowTitle(window, "N-Body Simulation (CPU)");
        }
        
        // --- LOGIC / PHYSICS UPDATE ---

        // KICK-DRIFT-KICK using the GPU for force calculation

        // Step 1: KICK (CPU) - Update velocity for the first half time step
        for (int i = 0; i < numParticles; ++i) {
            h_velX[i] += 0.5f * (h_forceX[i] / h_mass[i]) * dt;
            h_velY[i] += 0.5f * (h_forceY[i] / h_mass[i]) * dt;
        }

        // Step 2: DRIFT (CPU) - Update position for a full time step
        for (int i = 0; i < numParticles; ++i) {
            h_posX[i] += h_velX[i] * dt;
            h_posY[i] += h_velY[i] * dt;
        }

        // Step 3: RECALCULATE FORCES (GPU)
        if (use_gpu) {
            // Copy the latest positions to the GPU
            gpuErrchk(cudaMemcpy(d_posX, h_posX.data(), dataSize, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_posY, h_posY.data(), dataSize, cudaMemcpyHostToDevice));

            // Launch the kernel on the GPU
            int threadsPerBlock = 256;
            int numBlocks = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
            calculateForces<<<numBlocks, threadsPerBlock>>>(d_posX, d_posY, d_mass,
                                                            d_forceX, d_forceY,
                                                            numParticles, G, epsilon);

            // Check for errors after kernel launch
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize()); // Wait for the kernel to finish and check for errors

            // Copy the resulting forces back from the GPU to the CPU
            gpuErrchk(cudaMemcpy(h_forceX.data(), d_forceX, dataSize, cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_forceY.data(), d_forceY, dataSize, cudaMemcpyDeviceToHost));
        } else {
            // CPU force calculation (for comparison)
            // Reset forces
            for (int i = 0; i < numParticles; ++i) {
                h_forceX[i] = 0.0f;
                h_forceY[i] = 0.0f;
            }

            for (int i = 0; i < numParticles; ++i) {
                for (int j = 0; j < numParticles; ++j) {
                    if (i == j) continue;  // Skip self-interaction

                    // --- PLUMMER SOFTENING IMPLEMENTATION ---

                    // Vector from particle i to j
                    float dx = h_posX[j] - h_posX[i];
                    float dy = h_posY[j] - h_posY[i];

                    // True squared distance (r^2 from the formula)
                    float rSq = dx * dx + dy * dy;

                    // Denominator from formula (9.8): (r^2 + ε^2)^(3/2)
                    float denom = pow(rSq + epsilon * epsilon, 1.5);
                    
                    if (denom > 0) { // Avoid division by zero
                        // Calculate the force scalar part: F = G * m1 * m2 / denom
                        float force_scalar = (G * h_mass[i] * h_mass[j]) / denom;

                        // Add the force vector components to particle 'i'
                        h_forceX[i] += force_scalar * dx;
                        h_forceY[i] += force_scalar * dy;
                    }
                }
            }
        }

        // Step 4: KICK (CPU) - Update velocity for the second half time step
        for (int i = 0; i < numParticles; ++i) {
            h_velX[i] += 0.5f * (h_forceX[i] / h_mass[i]) * dt;
            h_velY[i] += 0.5f * (h_forceY[i] / h_mass[i]) * dt;
        }

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw all the particles as points
        glBegin(GL_POINTS);
        for (int i = 0; i < numParticles; ++i) {
            glVertex2f(h_posX[i], h_posY[i]);
        }
        glEnd();
        
        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // --- Cleanup ---
    glfwTerminate();
    gpuErrchk(cudaFree(d_posX));
    gpuErrchk(cudaFree(d_posY));
    gpuErrchk(cudaFree(d_mass));
    gpuErrchk(cudaFree(d_forceX));
    gpuErrchk(cudaFree(d_forceY));

    return 0;
}