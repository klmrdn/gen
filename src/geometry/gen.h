#pragma once

#include "vertex.h"

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <queue>
#include <random>
#include <unordered_set>

struct Sphere {
    glm::vec3 c;
    float r;

    Sphere(glm::vec3 c, float r) : c(c), r(r) {}

    bool operator==(const Sphere& rhs) const {
        return c == rhs.c && abs(r - rhs.r) < std::numeric_limits<float>::epsilon();
    }
};

namespace std {
    template<>
    struct hash<Sphere> {
        size_t operator()(Sphere const& s) const {
            return hash<glm::vec3>()(s.c) ^ (hash<float>()(s.r) << 1);
        }
    };
}

struct Triangle {
    glm::vec3 a, b, c;

    Triangle(glm::vec3 a, glm::vec3 b, glm::vec3 c) : a(a), b(b), c(c) {}
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    AABB() = default;
    AABB(glm::vec3 o, float delta) : min(glm::floor(o - delta)), max(glm::ceil(o + delta)) {}
    AABB(glm::vec3 min, glm::vec3 max) : min(glm::floor(min)), max(glm::ceil(max)) {}
    AABB(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, float r_max)
        : min(vertices.front().pos), max(vertices.front().pos) {
        for (const Vertex& vertex : vertices) {
            for (size_t i = 0; i < 3; ++i) {
                if (min[i] > vertex.pos[i]) min[i] = vertex.pos[i];
                if (max[i] < vertex.pos[i]) max[i] = vertex.pos[i];
            }
        }
        min -= r_max;
        max += r_max;

        min = glm::floor(min);
        max = glm::ceil(max);
    }

    glm::ivec3 index(glm::vec3 p, float d) { return glm::floor((p - min) / d); }
    glm::vec3 indexInvert(size_t i, size_t j, size_t k, float d) {
        return glm::vec3(i, j, k) * d + min + d / 2;
    }
};

struct Mesh {
    std::vector<Vertex>& vertices;
    std::vector<uint32_t>& indices;

    Mesh(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
        : vertices(vertices), indices(indices) {}

    Triangle get(int i) {
        return {vertices[indices[i]].pos, vertices[indices[i + 1]].pos, vertices[indices[i + 2]].pos};
    }

    glm::vec3 get_n(int i) {
        /*Triangle t = get(i);

        glm::vec3 n = glm::normalize(glm::cross(t.b - t.a, t.c - t.a));
        if (glm::dot(glm::normalize(vertices[indices[i]].pos), n) < 0.0f) {
            n = glm::normalize(glm::cross(t.b - t.c, t.a - t.c));
        }*/

        glm::vec3 n = glm::normalize(vertices[indices[i]].normal + vertices[indices[i + 1]].normal +
                                     vertices[indices[i + 2]].normal);


        return n;
    }
};

bool intersectLineTriangle(glm::vec3 p, glm::vec3 q, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    float u, v, w, t;

    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 qp = p - q;

    glm::vec3 n = glm::cross(ab, ac);

    float d = glm::dot(qp, n);
    if (d <= 0.0f) return false;

    glm::vec3 ap = p - a;
    t            = glm::dot(ap, n);
    if (t < 0.0f) return false;
    // if (t > d) return false;

    glm::vec3 e = glm::cross(qp, ap);
    v           = glm::dot(ac, e);
    if (v < 0.0f || v > d) return false;
    w = -glm::dot(ab, e);
    if (w < 0.0f || v + w > d) return false;

    return true;
}

int sgn(float x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

glm::vec3 random_on_unit_sphere() {
    std::default_random_engine gen;
    std::uniform_real_distribution<float> pdf(0.0f, 1.0f);

    double theta = 2.0f * std::numbers::pi * pdf(gen);
    double phi   = acos(1.0f - 2.0f * pdf(gen));

    return glm::vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
}

inline double det3x3(double a00, double a01, double a02, double a10, double a11, double a12, double a20,
                     double a21, double a22) {
    double m01 = a00 * a11 - a10 * a01;
    double m02 = a00 * a21 - a20 * a01;
    double m12 = a10 * a21 - a20 * a11;
    return m01 * a22 - m02 * a12 + m12 * a02;
}

inline int orient_3d_inexact(const glm::vec3 p0, const glm::vec3 p1, const glm::vec3 p2, const glm::vec3 p3) {
    double a11 = p1[0] - p0[0];
    double a12 = p1[1] - p0[1];
    double a13 = p1[2] - p0[2];

    double a21 = p2[0] - p0[0];
    double a22 = p2[1] - p0[1];
    double a23 = p2[2] - p0[2];

    double a31 = p3[0] - p0[0];
    double a32 = p3[1] - p0[1];
    double a33 = p3[2] - p0[2];

    double Delta = det3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33);

    return sgn(Delta);
}

// TODO dir should be constant for all tests
bool intersectLineTriangle2(glm::vec3 p, Triangle t) {
    int i = 0;
    while (true) {
        i++;
        if (i > 1000) { std::cout << "took to long to find value" << std::endl; }

        glm::vec3 dir = random_on_unit_sphere();
        dir           = glm::normalize(dir);
        glm::vec3 q   = p + dir * 10.0f;

        int s1 = orient_3d_inexact(p, t.a, t.b, t.c);
        if (s1 == 0) continue;
        int s2 = orient_3d_inexact(q, t.a, t.b, t.c);
        if (s2 == 0) continue;

        if (s1 == s2) { return false; }

        int s3 = orient_3d_inexact(p, q, t.a, t.b);
        if (s3 == 0) continue;
        int s4 = orient_3d_inexact(p, q, t.b, t.c);
        if (s4 == 0) continue;
        int s5 = orient_3d_inexact(p, q, t.c, t.a);
        if (s5 == 0) continue;

        return s3 == s4 && s4 == s5;
    }
}

glm::vec3 closestPointTriangle(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 ap = p - a;
    float d1     = glm::dot(ab, ap);
    float d2     = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    glm::vec3 bp = p - b;
    float d3     = glm::dot(ab, bp);
    float d4     = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    glm::vec3 cp = p - c;
    float d5     = glm::dot(ab, cp);
    float d6     = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    float denom = 1.0f / (va + vb + vc);
    float v     = vb * denom;
    float w     = vc * denom;
    return a + ab * v + ac * w;
}

bool convex(glm::vec3 a, glm::vec3 b) {
    float o = glm::angle(glm::normalize(a), glm::normalize(b));
    return o > 0.0 && o < std::numbers::pi;
}

float inter2d(float c00, float c10, float c01, float c11, float tx, float ty) {
    float a = c00 * (1 - tx) + c10 * tx;
    float b = c01 * (1 - tx) + c11 * tx;
    return a * (1 - ty) + b * ty;
}

float inter3d(float c000, float c100, float c010, float c110, float c001, float c101, float c011, float c111,
              float tx, float ty, float tz) {
    float e = inter2d(tx, ty, c000, c100, c010, c110);
    float f = inter2d(tx, ty, c001, c101, c011, c111);
    return e * (1 - tz) + f * tz;
}

struct Grid {
    AABB aabb;
    struct Point {
        float d = std::numeric_limits<float>::infinity();
        int ti  = -1;
        std::vector<Sphere> s;
    };
    std::vector<std::vector<std::vector<Point>>> grid;

    void insideMesh(Mesh& mesh, glm::ivec3 n, float r_max) {
        for (int i = 0; i < n.x; i++) {
            for (int j = 0; j < n.y; j++) {
                for (int k = 0; k < n.z; k++) {
                    // TODO
                    glm::vec3 start = aabb.indexInvert(i, j, k, r_max);

                    int c = 0;
                    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
                        Triangle t = mesh.get(i);
                        if (intersectLineTriangle2(start, t)) c++;
                    }

                    if (c % 2 == 1) {
                        grid[i][j][k] = Point{.d = -std::numeric_limits<float>::infinity()};
                    } else {
                        grid[i][j][k] = Point{.d = std::numeric_limits<float>::infinity()};
                    }
                }
            }
        }
    }

    void refine_grid(Mesh& mesh, float r_max) {
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            Triangle t  = mesh.get(i);
            glm::vec3 n = mesh.get_n(i);

            Triangle out(t.a + n * (1.5f * r_max + r_max), t.b + n * (1.5f * r_max + r_max),
                         t.c + n * (1.5f * r_max + r_max));
            Triangle in(t.a - n * (1.5f * r_max + r_max), t.b - n * (1.5f * r_max + r_max),
                        t.c - n * (1.5f * r_max + r_max));

            std::vector<glm::vec3> ps = {out.a, out.b, out.c, in.a, in.b, in.c};
            glm::vec3 min             = ps[0];
            glm::vec3 max             = ps[0];
            for (const glm::vec3& p : ps) {
                for (size_t i = 0; i < 3; ++i) {
                    if (min[i] > p[i]) min[i] = p[i];
                    if (max[i] < p[i]) max[i] = p[i];
                }
            }

            AABB local   = AABB(min, max);
            glm::ivec3 x = glm::ceil((local.max - local.min) / r_max);

            for (int gi = 0; gi < x.x; ++gi) {
                for (int gj = 0; gj < x.y; ++gj) {
                    for (int gk = 0; gk < x.z; ++gk) {
                        glm::vec3 start = local.indexInvert(gi, gj, gk, r_max);
                        if (start.x < aabb.min.x || start.y < aabb.min.y || start.z < aabb.min.z ||
                            start.x > aabb.max.x || start.y > aabb.max.y || start.z > aabb.max.z) {
                            continue;
                        }

                        glm::vec3 p = closestPointTriangle(start, t.a, t.b, t.c);
                        float d     = glm::distance(start, p);
                        if (glm::dot(glm::normalize(n), start - p) < 0.0f) { d *= -1; }

                        glm::ivec3 index = aabb.index(start, r_max);

                        Point& current = grid[index.x][index.y][index.z];

                        if (std::abs(d) < std::abs(current.d)) {
                            if (current.ti == -1 || sgn(d) == sgn(current.d)) {
                                current.d  = d;
                                current.ti = mesh.indices[i];

                                continue;
                            }

                            Triangle to  = mesh.get(current.ti);
                            glm::vec3 n2 = mesh.get_n(i);

                            if (convex(n, n2) && sgn(current.d) == 1 && sgn(d) == -1) {
                                current.d  = d;
                                current.ti = mesh.indices[i];
                            } else if (!convex(n, n2) && sgn(current.d) == -1 && sgn(d) == 1) {
                                current.d  = d;
                                current.ti = mesh.indices[i];
                            }
                        }
                    }
                }
            }
        }
    }

    Grid(Mesh mesh, float r_max) {
        aabb = AABB(mesh.vertices, mesh.indices, r_max);

        glm::ivec3 n = glm::ceil((aabb.max - aabb.min) / r_max);

        grid.resize(n.x);
        for (int i = 0; i < n.x; ++i) {
            grid[i].resize(n.y);
        }
        for (int i = 0; i < n.x; ++i) {
            for (int j = 0; j < n.y; ++j) {
                grid[i][j].resize(n.z);
            }
        }

        insideMesh(mesh, n, r_max);
        refine_grid(mesh, r_max);

        /*std::cout << std::endl << n.x << " " << n.y << " " << n.z << std::endl << std::endl;

        std::ofstream myfile;
        myfile.open("out.txt");
        for (size_t i = 0; i < n.x; i++) {
            for (size_t j = 0; j < n.y; j++) {
                for (size_t k = 0; k < n.z; k++) {
                    float d = grid[i][j][k].d;

                    myfile << d << " ";
                }
            }
        }
        myfile.close();*/
    }

    void push(Sphere s, float r_max) {
        glm::vec3 min = s.c - r_max;
        glm::vec3 max = s.c + r_max;
        AABB local(min, max);

        glm::ivec3 i = glm::ceil((local.max - local.min) / r_max);

        for (int gi = 0; gi < i.x; gi++) {
            for (int gj = 0; gj < i.y; gj++) {
                for (int gk = 0; gk < i.z; gk++) {
                    glm::vec3 p = local.indexInvert(gi, gj, gk, r_max);

                    if (p.x < aabb.min.x || p.y < aabb.min.y || p.z < aabb.min.z || p.x > aabb.max.x ||
                        p.y > aabb.max.y || p.z > aabb.max.z) {
                        continue;
                    }

                    glm::ivec3 index  = aabb.index(p, r_max);
                    Grid::Point& cell = grid[index.x][index.y][index.z];

                    cell.s.push_back(s);
                }
            }
        }
    }

    float distance(glm::vec3 p, float r_max) {
        glm::ivec3 index = aabb.index(p, r_max);

        if (index.x < 0 || index.y < 0 || index.z < 0 || index.x >= grid.size() ||
            index.y >= grid[0].size() || index.z >= grid[0][0].size() || index.x + 1 < 0 || index.y + 1 < 0 ||
            index.z + 1 < 0 || index.x + 1 >= grid.size() || index.y + 1 >= grid[0].size() ||
            index.z + 1 >= grid[0][0].size())
            return std::numeric_limits<float>::infinity();

        float c000 = grid[index.x][index.y][index.z].d;
        float c100 = grid[index.x + 1][index.y][index.z].d;
        float c010 = grid[index.x][index.y + 1][index.z].d;
        float c110 = grid[index.x + 1][index.y + 1][index.z].d;
        float c001 = grid[index.x][index.y][index.z + 1].d;
        float c101 = grid[index.x + 1][index.y][index.z + 1].d;
        float c011 = grid[index.x][index.y + 1][index.z + 1].d;
        float c111 = grid[index.x + 1][index.y + 1][index.z + 1].d;

        glm::vec3 pos = aabb.indexInvert(index.x, index.y, index.z, r_max);

        return inter3d(c000, c100, c010, c110, c001, c101, c011, c111, p.x - pos.x, p.y - pos.y, p.z - pos.z);
    }
};

float selectARadius(std::normal_distribution<float>& pdf, float r_min, float r_max,
                    std::deque<float>& prevRejectedQueue) {
    if (!prevRejectedQueue.empty()) {
        auto ret = prevRejectedQueue.front();
        prevRejectedQueue.pop_front();
        return ret;
    }

    std::default_random_engine gen;

    while (true) {
        float n = pdf(gen);
        if (n >= r_min && n <= r_max) return n;
    }
}

AABB computeNeighborhoodBox(Sphere& currentSphere, float r_new, float r_max) {
    return AABB(currentSphere.c, currentSphere.r + 2.0f * r_new);
}


std::vector<Sphere> gatherSpheresThatIntersect(Grid gridIndex, AABB box, float r_max) {
    std::unordered_set<Sphere> spheres;

    int x = std::ceil((box.max.x - box.min.x) / r_max);
    int y = std::ceil((box.max.y - box.min.y) / r_max);
    int z = std::ceil((box.max.z - box.min.z) / r_max);

    for (int gi = 0; gi < x; ++gi) {
        for (int gj = 0; gj < y; ++gj) {
            for (int gk = 0; gk < z; ++gk) {
                glm::vec3 p = box.indexInvert(gi, gj, gk, r_max);

                if (p.x < gridIndex.aabb.min.x || p.y < gridIndex.aabb.min.y || p.z < gridIndex.aabb.min.z ||
                    p.x >= gridIndex.aabb.max.x || p.y >= gridIndex.aabb.max.y ||
                    p.z >= gridIndex.aabb.max.z) {
                    continue;
                }

                glm::ivec3 index  = gridIndex.aabb.index(p, r_max);
                Grid::Point& cell = gridIndex.grid[index.x][index.y][index.z];

                std::copy(cell.s.begin(), cell.s.end(), std::inserter(spheres, spheres.end()));
            }
        }
    }

    std::vector<Sphere> out(spheres.begin(), spheres.end());
    return out;
}

glm::vec3 computeC(glm::vec3 c1, float r1, glm::vec3 c2, float r2) {
    float d = glm::distance(c1, c2);
    float a = 0.5f * (1.0f - (r2 * r2 - r1 * r1) / (d * d));

    return (1.0f - a) * c1 + a * c2;
}

float computeR(glm::vec3 c1, float r1, glm::vec3 cc) {
    float d = glm::distance(c1, cc);
    float r = std::sqrt(r1 * r1 - d * d);

    if (isnan(r)) {
        return 0.0f;
        throw std::runtime_error("return value nan!");
    }

    return r;
}

std::vector<glm::vec3> gatherAllInterceptions(Sphere s, std::vector<Sphere> others, float r_new) {
    std::vector<glm::vec3> ps;

    for (const Sphere& lhs : others) {
        if (lhs == s) continue;
        for (const Sphere& rhs : others) {
            if (rhs == lhs || rhs == s) continue;

            float r_lhs = lhs.r + r_new;
            float r_rhs = rhs.r + r_new;
            float r_cur = s.r + r_new;

            float tt4 = r_lhs + r_rhs;
            float tt5 = glm::distance(lhs.c, rhs.c);
            if (tt4 <= tt5) continue;

            glm::vec3 n  = glm::normalize(rhs.c - lhs.c);
            glm::vec3 cc = computeC(lhs.c, r_lhs, rhs.c, r_rhs);
            float rc     = computeR(lhs.c, r_lhs, cc);

            float dd = glm::dot(n, cc - s.c);
            if (abs(dd) > r_cur) continue;

            glm::vec3 tt3 = s.c - cc;
            float l       = glm::dot((s.c - cc), n);

            glm::vec3 cp = s.c - l * n;
            float rp     = std::sqrt(r_cur * r_cur - l * l);

            auto tmp = cp - cc;
            float o  = rp + rc - glm::length(cp - cc);

            float tt1 = glm::distance(cc, cp);
            float tt2 = abs(rp - rc);
            if (o <= 0.0f || tt1 < tt2) continue;

            glm::vec3 cm = computeC(cc, rc, cp, rp);
            float rm     = computeR(cc, rc, cm);

            glm::vec3 p1 = cm + rm * glm::normalize(glm::cross(n, cc - cm));
            glm::vec3 p2 = cm - rm * glm::normalize(glm::cross(n, cc - cm));

            if (isnan(p1.x) || isnan(p1.y) || isnan(p1.z) || isnan(p2.x) || isnan(p2.y) || isnan(p2.z)) {
                throw std::runtime_error("return value nan!");
            }

            ps.push_back(p1);
            ps.push_back(p2);
        }
    }

    return ps;
}

glm::vec3 bestPointIn(std::vector<glm::vec3> ps, glm::vec3 seed) {
    auto result = std::min_element(ps.begin(), ps.end(), [&seed](glm::vec3 a, glm::vec3 b) {
        return glm::distance(seed, a) < glm::distance(seed, b);
    });
    return *result;
}

std::vector<glm::vec3> gen(Mesh mesh, float r_min, float r_max, std::normal_distribution<float>& pdf,
                           glm::vec3 seed_position) {
    std::cout << "start" << std::endl;
    std::queue<Sphere> frontSpheresQueue;
    std::vector<Sphere> assemblyList;
    Grid gridIndex = Grid(mesh, r_max);
    std::deque<float> prevRejectedQueue;
    std::deque<float> newlyRejectedQueue;
    std::cout << "done setup" << std::endl;

    for (size_t i = 0; i < 3; i++) {
        float theta = 2.0f * std::numbers::pi * float(i) / 3.0f;

        glm::vec3 c = {r_max * cos(theta) + seed_position.x, r_max * sin(theta) + seed_position.y,
                       seed_position.z};

        Sphere a(c, r_max);
        frontSpheresQueue.push(a);
        assemblyList.push_back(a);
        gridIndex.push(a, r_max);
    }

    while (!frontSpheresQueue.empty()) {
        std::cout << "step " << frontSpheresQueue.size() << std::endl;
        Sphere currentSphere    = frontSpheresQueue.front();
        float r_new             = selectARadius(pdf, r_min, r_max, prevRejectedQueue);
        AABB box                = computeNeighborhoodBox(currentSphere, r_new, r_max);
        auto neighboringSpheres = gatherSpheresThatIntersect(gridIndex, box, r_max);
        auto candidatePointList = gatherAllInterceptions(currentSphere, neighboringSpheres, r_new);

        // removeAllThatIntercepts(candidatePointList, gridIndex, r_max);
        auto toRemove1 = std::remove_if(
          candidatePointList.begin(), candidatePointList.end(), [&gridIndex, &mesh, r_max](glm::vec3 p) {
              // bool ret = gridIndex.distance(p, r_max) >= 0.0f;
              if (p.x < gridIndex.aabb.min.x || p.y < gridIndex.aabb.min.y || p.z < gridIndex.aabb.min.z ||
                  p.x >= gridIndex.aabb.max.x || p.y >= gridIndex.aabb.max.y || p.z >= gridIndex.aabb.max.z) {
                  return true;
              }

              glm::ivec3 index = gridIndex.aabb.index(p, r_max);
              bool ret         = gridIndex.grid[index.x][index.y][index.z].d >= 0;

              //{
              //    glm::vec3 start = p;
              //    int c           = 0;
              //    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
              //        Triangle t(mesh.vertices[mesh.indices[i]].pos, mesh.vertices[mesh.indices[i + 1]].pos,
              //                   mesh.vertices[mesh.indices[i + 2]].pos);
              //        if (intersectLineTriangle2(start, t)) { c++; }
              //    }
              //    if ((c % 2 == 0) != ret) {//
              //        std::cout << "differnce in removeif" << std::endl;
              //    }
              //}

              return ret;
          });
        candidatePointList.erase(toRemove1, candidatePointList.end());

        // removeAllThatIntercepts(candidatePointList, neighboringSpheres, r_new);
        auto toRemove2 = std::remove_if(candidatePointList.begin(), candidatePointList.end(),
                                        [&neighboringSpheres, &r_new](glm::vec3 p) {
                                            for (const Sphere& s : neighboringSpheres) {
                                                float d  = glm::distance(s.c, p);
                                                float r  = (s.r + r_new);
                                                float tt = abs(d - r);

                                                bool tt2 = d < r;
                                                bool tt3 = tt > 0.001f;
                                                if (tt2) { return true; }
                                            }
                                            return false;
                                        });
        candidatePointList.erase(toRemove2, candidatePointList.end());

        size_t numberoFValidPoints = candidatePointList.size();
        if (numberoFValidPoints > 0) {
            glm::vec3 bestPoint = bestPointIn(candidatePointList, seed_position);
            auto newSphere      = Sphere(bestPoint, r_new);
            frontSpheresQueue.push(newSphere);
            assemblyList.push_back(newSphere);
            gridIndex.push(newSphere, r_max);

            prevRejectedQueue.insert(prevRejectedQueue.end(),
                                     std::make_move_iterator(newlyRejectedQueue.begin()),
                                     std::make_move_iterator(newlyRejectedQueue.end()));
            newlyRejectedQueue.clear();
        } else {
            newlyRejectedQueue.push_back(r_new);
        }
        if (numberoFValidPoints < 2) { frontSpheresQueue.pop(); }
    }

    std::vector<glm::vec3> ps;
    std::transform(assemblyList.begin(), assemblyList.end(), std::back_inserter(ps),
                   [](Sphere s) { return s.c; });

    std::cout << "end" << std::endl;

    return ps;
}
