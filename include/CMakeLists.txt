include(FetchContent)

find_package(Vulkan REQUIRED FATAL_ERROR)

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    fetch_glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
    GIT_TAG        3.3.6 
)
FetchContent_MakeAvailable(fetch_glfw)

# glm
FetchContent_Declare(
    fetch_glm
    GIT_REPOSITORY https://github.com/g-truc/glm
    GIT_TAG        0.9.9.8 
)
FetchContent_MakeAvailable(fetch_glm)

# imgui
FetchContent_Declare(
    fetch_imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG        v1.87
)
FetchContent_MakeAvailable(fetch_imgui)

add_library(imgui STATIC)
target_include_directories(imgui PUBLIC
    ${fetch_imgui_SOURCE_DIR}
    ${fetch_imgui_SOURCE_DIR}/backends
    ${fetch_imgui_SOURCE_DIR}/include
)
target_sources(imgui PRIVATE 
    ${fetch_imgui_SOURCE_DIR}/imgui.cpp

    ${fetch_imgui_SOURCE_DIR}/imgui_demo.cpp
    ${fetch_imgui_SOURCE_DIR}/imgui_draw.cpp
    ${fetch_imgui_SOURCE_DIR}/imgui_tables.cpp
    ${fetch_imgui_SOURCE_DIR}/imgui_widgets.cpp

    ${fetch_imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${fetch_imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

# stb_image
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE stb_image)

# tinyibjloader
FetchContent_Declare(
    fetch_tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader
    GIT_TAG        8322e00ae685ea623ab6ac5a6cebcfa2d22fbf93 
)
FetchContent_MakeAvailable(fetch_tinyobjloader)

# vma
FetchContent_Declare(
    fetch_vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG        042e65e7c1bd7cfc759688b6e527cd7a96e837ce 
)
FetchContent_MakeAvailable(fetch_vma)

# SPIRV-Reflect
add_library(spirv_reflect INTERFACE)
target_include_directories(spirv_reflect INTERFACE stb_image)

set_target_properties(imgui PROPERTIES FOLDER "include")
set_target_properties(tinyobjloader PROPERTIES FOLDER "include")
set_target_properties(VulkanMemoryAllocator PROPERTIES FOLDER "include")
