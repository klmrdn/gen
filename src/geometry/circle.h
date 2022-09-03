#pragma once

#include "vk/vk_types.h"

#include "vertex.h"

#include <numbers>
#include <vector>

const int SEGMENTS = 10;

class Circle {
public:
    float cx, cy, r;

    std::vector<Vertex> vertices;
    AllocatedBuffer vertexBuffer;

    void set1(const VmaAllocator& allocator, const VkDevice& device, const VkCommandPool& commandPool,
              const VkQueue& graphicsQueue) {
        for (size_t i = 0; i < SEGMENTS; i++) {
            float theta = 2.0f * std::numbers::pi * float(i) / float(SEGMENTS);

            Vertex vertex{};
            vertex.pos      = {r * cos(theta) + cx, r * sin(theta) + cy, 0.0f};
            vertex.texCoord = {0, 0};
            vertex.color    = {1.0f, 1.0f, 1.0f};
            vertices.push_back(vertex);
        }
        vertices.push_back(vertices[0]);
    }

    void set2(const VmaAllocator& allocator, const VkDevice& device, const VkCommandPool& commandPool,
              const VkQueue& graphicsQueue) {
        createVertexBuffer(allocator, device, commandPool, graphicsQueue, vertices, vertexBuffer);
    }

    void recordCommandBuffer(VkCommandBuffer cmd) {
        VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
        VkDeviceSize offsets[]   = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

        vkCmdDraw(cmd, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
    }

    void free(VmaAllocator allocator) {
        vmaDestroyBuffer(allocator, vertexBuffer.buffer, vertexBuffer.allocation);
    }
};
