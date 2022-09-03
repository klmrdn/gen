find_package(Vulkan REQUIRED FATAL_ERROR)

add_executable(gen
    main.cpp
    
    vk/vk_types.h
    vk/vk_debug.h vk/vk_debug.cpp
    vk/vk_engine.h vk/vk_engine.cpp

    geometry/dewall.h geometry/dewall.cpp
    geometry/inter.h
    geometry/gen.h geometry/gen.cpp
    geometry/vertex.h geometry/vertex.cpp
    geometry/circle.h geometry/circle.cpp

    stb_image.cpp
    tiny_obj_loader.cpp
)

target_link_libraries(gen PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    imgui
    spirv_reflect
    stb_image
    tinyobjloader
    VulkanMemoryAllocator
)

add_dependencies(gen shaders)

add_custom_command(
    TARGET gen POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        ${CMAKE_BINARY_DIR}/models
)
add_custom_command(
    TARGET gen POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/textures
        ${CMAKE_BINARY_DIR}/textures
)
