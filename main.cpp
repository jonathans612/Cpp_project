#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>  // Defines std::vector
#include <cstdlib> // Defines rand() and srand()
#include <ctime>   // Defines time()

const float G = 1.0;      // 6.674e-5 // A "tuned" gravitational constant for our simulation
const float a = 10.0;     // Softening factor to prevent division by zero
const int numParticles = 3; // Number of particles in the simulation

// A single particle in our simulation
struct Particle {
    float posX, posY; // Position
    float velX, velY; // Velocity
    float forceX, forceY; // Total force acting on the particle
    float mass;

    // Constructor
    Particle(int screenWidth, int screenHeight) {
        posX = rand() % screenWidth;
        posY = rand() % screenHeight;
        velX = 0.0f;
        velY = 0.0f;
        // velX = ((rand() % 100) / 50.0f) - 1.0f; // Give a small random velocity
        // velY = ((rand() % 100) / 50.0f) - 1.0f;
        forceX = 0.0f;
        forceY = 0.0f;
        mass = (rand() % 1000 / 100.0f) + 1.0f; // Random mass between 1.0 and 11.0
    }

    // Reset the force for the next frame of calculation
    void resetForce() {
        forceX = 0.0f;
        forceY = 0.0f;
    }

    // Update the particle's velocity and position based on the calculated force
    void update() {
        // Update velocity based on force and mass (a = F/m)
        float accelX = forceX / mass;
        float accelY = forceY / mass;
        velX += accelX;
        velY += accelY;

        // Update position based on velocity
        posX += velX;
        posY += velY;
    }

    // Draw the particle as a single point
    void draw() {
        glVertex2f(posX, posY);
    }
};

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
    srand(time(0));
    std::vector<Particle> particles;

    for (int i = 0; i < numParticles; ++i) {
        particles.push_back(Particle(width, height)); // Use the actual window width/height
    }

    // Main loop: runs until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // --- INPUT HANDLING ---
        // Check if the ESC key is pressed and set the window to close
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        
        // --- LOGIC / PHYSICS UPDATE ---

        // Step A: Reset forces for all particles
        for (auto& p : particles) {
            p.resetForce();
        }

        // Step B: Calculate gravitational forces between every pair of particles
        for (int i = 0; i < numParticles; ++i) {
            for (int j = 0; j < numParticles; ++j) {
                if (i == j) continue; // A particle doesn't attract itself

                float dx = particles[j].posX - particles[i].posX;
                float dy = particles[j].posY - particles[i].posY;
                float distSq = dx * dx + dy * dy + a * a; // Squared distance with softening
                float dist = sqrt(distSq);

                // Calculate force magnitude
                float force = (G * particles[i].mass * particles[j].mass) / distSq;

                // Add to the total force acting on particle 'i'
                particles[i].forceX += force * dx / dist;
                particles[i].forceY += force * dy / dist;
            }
        }

        // Step C: Update particle positions based on the calculated forces
        for (auto& p : particles) {
            p.update();
        }

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw all the particles as points
        glBegin(GL_POINTS);
        for (auto& p : particles) {
            p.draw();
        }
        glEnd();
        
        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}