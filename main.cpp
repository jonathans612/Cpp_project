#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>  // Defines std::vector
#include <cstdlib> // Defines rand() and srand()
#include <ctime>   // Defines time()

// Define the blueprint for a single particle
struct Particle {
    float posX, posY; // Position
    float velX, velY; // Velocity

    // Constructor to initialize a particle with a random position and velocity
    Particle(int screenWidth, int screenHeight) {
        // Start at a random position within the screen bounds
        posX = rand() % screenWidth;
        posY = rand() % screenHeight;

        // Start with a random velocity
        velX = ((rand() % 100) / 50.0f) - 1.0f; // Random float between -1.0 and 1.0
        velY = ((rand() % 100) / 50.0f) - 1.0f; // Random float between -1.0 and 1.0
    }

    // Update the particle's position based on its velocity
    void update() {
        posX += velX;
        posY += velY;
    }

    // Draw the particle as a single point
    void draw() {
        glVertex2f(posX, posY); // Specify the vertex for our point
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
    const int numParticles = 1000;

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
        // Update the position of every particle
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