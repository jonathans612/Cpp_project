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
const float G = 1.0f;
const float epsilon = 10.0f;
const int numParticles = 6;  // Number of particles in the simulation
const float dt = 1.0f;       // Our time step

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

    // --- Host (CPU) Data Structures ---
    srand(time(0));
    size_t dataSize = numParticles * sizeof(float);
    std::vector<float> h_posX(numParticles);
    std::vector<float> h_posY(numParticles);
    std::vector<float> h_velX(numParticles, 0.0f); // Initialize velocities to 0
    std::vector<float> h_velY(numParticles, 0.0f);
    std::vector<float> h_forceX(numParticles, 0.0f);
    std::vector<float> h_forceY(numParticles, 0.0f);
    std::vector<float> h_mass(numParticles);

    for (int i = 0; i < numParticles; ++i) {
        h_posX[i] = rand() % width;
        h_posY[i] = rand() % height;
        h_mass[i] = (rand() % 1000 / 100.0f) + 1.0f;  // Random mass between 1.0 and 11.0
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

    // Main loop: runs until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // --- INPUT HANDLING ---
        // Check if the ESC key is pressed and set the window to close
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
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