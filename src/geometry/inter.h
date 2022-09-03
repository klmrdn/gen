#pragma once

#include "dewall.h"

#include <vector>

double parallelogram_area_3d(glm::vec3 p[4]) {
    glm::vec3 cross;
    cross[0] = (p[1][1] - p[0][1]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1]);
    cross[1] = (p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]);
    cross[2] = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);

    return glm::length(cross);
}


float quad_area_3d(std::vector<glm::vec3> q) {
    glm::vec3 p[4];
    p[0] = 0.5f * (q[0] + q[1]);
    p[1] = 0.5f * (q[1] + q[2]);
    p[2] = 0.5f * (q[2] + q[3]);
    p[3] = 0.5f * (q[3] + q[0]);

    return 2.0 * parallelogram_area_3d(p);
}

void plane_normal_tetrahedron_intersect(glm::vec3 pp, glm::vec3 normal, dewall::_Tetra t,
                                        const std::vector<glm::vec3>& global_point_set,
                                        std::vector<glm::vec3>& pint) {
    normal     = glm::normalize(normal);
    double dpp = glm::dot(normal, pp);

    // d[i] is positive, zero, or negative if vertex i is above, on, or below the
    // plane.
    double d[4];
    for (int i = 0; i < 4; i++) {
        d[i] = glm::dot(normal, global_point_set[t[i]]) - dpp;
    }

    // If all d are positive or negative, no intersection.
    if ((d[0] < 0.0f && d[1] < 0.0f && d[2] < 0.0f && d[3] < 0.0f) ||
        (d[0] > 0.0f && d[1] > 0.0f && d[2] > 0.0f && d[3] > 0.0f)) {
        return;
    }

    // Points with zero distance are automatically added to the list.
    //
    // For each point with nonzero distance, seek another point with opposite sign
    // and higher index, and compute the intersection of the line between those
    // points and the plane.
    for (int i = 0; i < 4; i++) {
        if (std::abs(d[i]) < 1.0e-8f) {
            pint.push_back(global_point_set[t[i]]);
        } else {
            for (int j = i + 1; j < 4; j++) {
                if (d[i] * d[j] < 0.0f) {
                    float inv_dist = 1.0f / (d[i] - d[j]);
                    glm::vec3 p;
                    for (int k = 0; k < 3; k++) {
                        p[k] =
                          (d[i] * global_point_set[t[j]][k] - d[j] * global_point_set[t[i]][k]) * inv_dist;
                    }
                    pint.push_back(p);
                }
            }
        }
    }

    //  If four points were found, order them properly.
    if (pint.size() == 4) {
        float area1 = quad_area_3d(pint);
        glm::vec3 pint2[4];

        pint2[0] = pint[0];
        pint2[1] = pint[2];

        pint2[2] = pint[3];
        pint2[3] = pint[2];

        double area2 = quad_area_3d(std::vector(pint2, pint2 + 4));

        if (area1 < area2) {
            for (size_t i = 0; i < 4; i++) {
                pint[i] = pint2[i];
            }
        }
    }

    return;
}

void batch_intersect_tris(glm::vec3 pp, glm::vec3 normal, std::vector<dewall::_Tetra> Tet_list,
                          const std::vector<glm::vec3>& global_point_set,
                          std::vector<std::vector<glm::vec3>>& pint) {
    for (int i = 0; i < Tet_list.size(); i++) {
        std::vector<glm::vec3> a;
        plane_normal_tetrahedron_intersect(pp, normal, Tet_list[i], global_point_set, a);

        if (a.size() == 3) pint.push_back(a);
        if (a.size() == 4) {}
    }
}