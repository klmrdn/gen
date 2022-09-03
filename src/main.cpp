#include "geometry/dewall.h"
#include "vk/vk_engine.h"

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <iostream>
#include <vector>

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
