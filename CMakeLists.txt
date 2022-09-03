cmake_minimum_required(VERSION 3.21)
project(gen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY VS_STARTUP_PROJECT gen)

add_subdirectory(include)
add_subdirectory(shaders)
add_subdirectory(src)
