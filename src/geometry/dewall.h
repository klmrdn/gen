#pragma once

#include "vertex.h"

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/vector_angle.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#define EPSILON 0.0000001

namespace dewall {
    struct Plane {
        glm::vec3 normal;
        glm::vec3 position;

        Plane(const glm::vec3& normal, const glm::vec3& position) : normal(normal), position(position) {}
        Plane(std::vector<glm::vec3> point_set, glm::vec3 min, glm::vec3 max) {
            auto sub = max - min;
            if (sub.x >= sub.z) {
                if (sub.y >= sub.x) {
                    normal = glm::vec3(0.0f, 1.0f, 0.0f);
                } else {
                    normal = glm::vec3(1.0f, 0.0f, 0.0f);
                }
            } else {
                if (sub.y >= sub.z) {
                    normal = glm::vec3(0.0f, 1.0f, 0.0f);
                } else {
                    normal = glm::vec3(0.0f, 0.0f, 1.0f);
                }
            }

            position = (max + min) / 2.0f;
        }

        float distance(const glm::vec3& point) const {
            return std::abs(glm::dot((point - position), normal));
        }

        bool intersect(const glm::vec3& start, const glm::vec3& direction) {
            float d = glm::dot(normal, direction);
            float n = -glm::dot(normal, start - position);

            if (std::abs(d) < EPSILON) return false;

            float t = n / d;
            if (t < 0 || t > 1) return false;

            return true;
        }
    };

    // haromszogbe kore irhato kor sugara
    float get_circum_circle_radius(const float distace_ab, const float distace_bc, const float distace_ac) {
        return ((distace_ab * distace_bc * distace_ac) /
                std::sqrt(((distace_ab + distace_bc + distace_ac) * (distace_bc + distace_ac - distace_ab) *
                           (distace_ac + distace_ab - distace_bc) * (distace_ab + distace_bc - distace_ac))));
    }

    // tetraeder kore irhato gomb sugara
    float get_circum_sphere_radius(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {

        double u   = (a.z - b.z) * (c.x * d.y - d.x * c.y) - (b.z - c.z) * (d.x * a.y - a.x * d.y);
        double v   = (c.z - d.z) * (a.x * b.y - b.x * a.y) - (d.z - a.z) * (b.x * c.y - c.x * b.y);
        double w   = (a.z - c.z) * (d.x * b.y - b.x * d.y) - (b.z - d.z) * (a.x * c.y - c.x * a.y);
        double uvw = 2 * (u + v + w);
        if (uvw == 0.0) {//
            throw std::runtime_error("coplanar");
        }
        double ra = glm::length2(a);
        double rb = glm::length2(b);
        double rc = glm::length2(c);
        double rd = glm::length2(d);
        double x0 = ((ra * (b.y * (c.z - d.z) + c.y * (d.z - b.z) + d.y * (b.z - c.z)) -
                      rb * (c.y * (d.z - a.z) + d.y * (a.z - c.z) + a.y * (c.z - d.z)) +
                      rc * (d.y * (a.z - b.z) + a.y * (b.z - d.z) + b.y * (d.z - a.z)) -
                      rd * (a.y * (b.z - c.z) + b.y * (c.z - a.z) + c.y * (a.z - b.z))) /
                     uvw);
        double y0 = ((ra * (b.z * (c.x - d.x) + c.z * (d.x - b.x) + d.z * (b.x - c.x)) -
                      rb * (c.z * (d.x - a.x) + d.z * (a.x - c.x) + a.z * (c.x - d.x)) +
                      rc * (d.z * (a.x - b.x) + a.z * (b.x - d.x) + b.z * (d.x - a.x)) -
                      rd * (a.z * (b.x - c.x) + b.z * (c.x - a.x) + c.z * (a.x - b.x))) /
                     uvw);
        double z0 = ((ra * (b.x * (c.y - d.y) + c.x * (d.y - b.y) + d.x * (b.y - c.y)) -
                      rb * (c.x * (d.y - a.y) + d.x * (a.y - c.y) + a.x * (c.y - d.y)) +
                      rc * (d.x * (a.y - b.y) + a.x * (b.y - d.y) + b.x * (d.y - a.y)) -
                      rd * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))) /
                     uvw);
        return glm::length(glm::vec3(a.x - x0, a.y - y0, a.z - z0));
    }

    // tetraeder kore irhato gomb kozeppontja
    glm::vec3 get_circum_sphere_center(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {

        double u   = (a.z - b.z) * (c.x * d.y - d.x * c.y) - (b.z - c.z) * (d.x * a.y - a.x * d.y);
        double v   = (c.z - d.z) * (a.x * b.y - b.x * a.y) - (d.z - a.z) * (b.x * c.y - c.x * b.y);
        double w   = (a.z - c.z) * (d.x * b.y - b.x * d.y) - (b.z - d.z) * (a.x * c.y - c.x * a.y);
        double uvw = 2 * (u + v + w);
        if (uvw == 0.0) {//
            throw std::runtime_error("coplanar");
        }
        double ra = glm::length2(a);
        double rb = glm::length2(b);
        double rc = glm::length2(c);
        double rd = glm::length2(d);
        double x0 = ((ra * (b.y * (c.z - d.z) + c.y * (d.z - b.z) + d.y * (b.z - c.z)) -
                      rb * (c.y * (d.z - a.z) + d.y * (a.z - c.z) + a.y * (c.z - d.z)) +
                      rc * (d.y * (a.z - b.z) + a.y * (b.z - d.z) + b.y * (d.z - a.z)) -
                      rd * (a.y * (b.z - c.z) + b.y * (c.z - a.z) + c.y * (a.z - b.z))) /
                     uvw);
        double y0 = ((ra * (b.z * (c.x - d.x) + c.z * (d.x - b.x) + d.z * (b.x - c.x)) -
                      rb * (c.z * (d.x - a.x) + d.z * (a.x - c.x) + a.z * (c.x - d.x)) +
                      rc * (d.z * (a.x - b.x) + a.z * (b.x - d.x) + b.z * (d.x - a.x)) -
                      rd * (a.z * (b.x - c.x) + b.z * (c.x - a.x) + c.z * (a.x - b.x))) /
                     uvw);
        double z0 = ((ra * (b.x * (c.y - d.y) + c.x * (d.y - b.y) + d.x * (b.y - c.y)) -
                      rb * (c.x * (d.y - a.y) + d.x * (a.y - c.y) + a.x * (c.y - d.y)) +
                      rc * (d.x * (a.y - b.y) + a.x * (b.y - d.y) + b.x * (d.y - a.y)) -
                      rd * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))) /
                     uvw);
        return glm::vec3(x0, y0, z0);
    }

    struct _Triangle {
        int a;
        int b;
        int c;
        int opposite;

        _Triangle(int a, int b, int c, int opposite = -1) : a(a), b(b), c(c), opposite(opposite) {
            if (a == b || a == c || b == c) {//
                throw std::runtime_error("invalid triangle");
            }
        }

        bool operator==(const _Triangle& rhs) const {
            return a == rhs.a && b == rhs.b && c == rhs.c ||//
                   a == rhs.a && b == rhs.c && c == rhs.b ||//

                   a == rhs.b && b == rhs.a && c == rhs.c ||//
                   a == rhs.b && b == rhs.c && c == rhs.a ||//

                   a == rhs.c && b == rhs.a && c == rhs.b ||//
                   a == rhs.c && b == rhs.b && c == rhs.a;
        }
        bool operator!=(const _Triangle& rhs) const { return !(*this == rhs); }
    };

    struct _Tetra {
        int a, b, c, d;

        _Tetra() = default;
        _Tetra(int a, int b, int c, int d) : a(a), b(b), c(c), d(d) {
            if (a == b || a == c || a == d || b == c || b == d || c == d) {//
                throw std::runtime_error("invalid tetra");
            }
        }

        bool operator==(const _Tetra& rhs) const {
            return a == rhs.a && b == rhs.b && c == rhs.c && d == rhs.d ||//
                   a == rhs.a && b == rhs.b && c == rhs.d && d == rhs.c ||//
                   a == rhs.a && b == rhs.c && c == rhs.b && d == rhs.d ||//
                   a == rhs.a && b == rhs.c && c == rhs.d && d == rhs.b ||//
                   a == rhs.a && b == rhs.d && c == rhs.b && d == rhs.c ||//
                   a == rhs.a && b == rhs.d && c == rhs.c && d == rhs.b ||//

                   a == rhs.b && b == rhs.a && c == rhs.c && d == rhs.d ||//
                   a == rhs.b && b == rhs.a && c == rhs.d && d == rhs.c ||//
                   a == rhs.b && b == rhs.c && c == rhs.a && d == rhs.d ||//
                   a == rhs.b && b == rhs.c && c == rhs.d && d == rhs.a ||//
                   a == rhs.b && b == rhs.d && c == rhs.a && d == rhs.c ||//
                   a == rhs.b && b == rhs.d && c == rhs.c && d == rhs.a ||//


                   a == rhs.c && b == rhs.a && c == rhs.b && d == rhs.d ||//
                   a == rhs.c && b == rhs.a && c == rhs.d && d == rhs.b ||//
                   a == rhs.c && b == rhs.b && c == rhs.a && d == rhs.d ||//
                   a == rhs.c && b == rhs.b && c == rhs.d && d == rhs.a ||//
                   a == rhs.c && b == rhs.d && c == rhs.a && d == rhs.b ||//
                   a == rhs.c && b == rhs.d && c == rhs.b && d == rhs.a ||//

                   a == rhs.d && b == rhs.a && c == rhs.b && d == rhs.c ||//
                   a == rhs.d && b == rhs.a && c == rhs.c && d == rhs.b ||//
                   a == rhs.d && b == rhs.b && c == rhs.a && d == rhs.c ||//
                   a == rhs.d && b == rhs.b && c == rhs.c && d == rhs.a ||//
                   a == rhs.d && b == rhs.c && c == rhs.a && d == rhs.b ||//
                   a == rhs.d && b == rhs.c && c == rhs.b && d == rhs.a;
        }
        bool operator!=(const _Tetra& rhs) const { return !(*this == rhs); }

        int operator[](int index) const {
            switch (index) {
            case 0:
                return a;
                break;
            case 1:
                return b;
                break;
            case 2:
                return c;
                break;
            case 3:
                return d;
                break;
            default:
                throw std::runtime_error("worng index");
                break;
            }
        }

        _Triangle abc() const { return _Triangle(a, b, c, d); }
        _Triangle bcd() const { return _Triangle(b, c, d, a); }
        _Triangle cda() const { return _Triangle(c, d, a, b); }
        _Triangle dab() const { return _Triangle(d, a, b, c); }
    };

    void partition_point_set(std::vector<glm::vec3> point_set, Plane wall, std::vector<glm::vec3>& p1,
                             std::vector<glm::vec3>& p2) {
        for (auto& point : point_set) {
            if (glm::dot(point - wall.position, wall.normal) >= 0.0f)
                p2.push_back(point);
            else
                p1.push_back(point);
        }
    }

    _Tetra make_first_simplex(const std::vector<glm::vec3>& point_set, Plane wall) {
        // First point
        int a             = -1;
        float distance_aw = 0.0f;
        for (int i = 0; i < point_set.size(); i++) {
            float distance = wall.distance(point_set[i]);
            if (distance < distance_aw || a == -1) {
                a           = i;
                distance_aw = distance;
            }
        }

        // Second point
        int b             = -1;
        float distance_ab = 0.0f;
        for (int i = 0; i < point_set.size(); i++) {
            glm::vec3 direction = point_set[i] - point_set[a];

            if (!wall.intersect(point_set[a], direction) || i == a) continue;

            float distance = glm::length2(direction);
            if (distance < distance_ab || b == -1) {
                b           = i;
                distance_ab = distance;
            }
        }

        // Third point
        int c                      = -1;
        float circum_circle_radius = 0.0f;
        for (int i = 0; i < point_set.size(); i++) {
            if (i == a || i == b) continue;

            float radius =
              get_circum_circle_radius(std::sqrt(distance_ab), glm::length(point_set[i] - point_set[b]),
                                       glm::length(point_set[i] - point_set[a]));
            if (radius < circum_circle_radius || c == -1) {
                c                    = i;
                circum_circle_radius = radius;
            }
        }

        int d                      = -1;
        float circum_sphere_radius = 0.0f;
        for (int i = 0; i < point_set.size(); i++) {
            if (i == a || i == b || i == c) continue;

            float radius = get_circum_sphere_radius(point_set[a], point_set[b], point_set[c], point_set[i]);
            if (radius < circum_sphere_radius || d == -1) {
                d                    = i;
                circum_sphere_radius = radius;
            }
        }

        return _Tetra(a, b, c, d);
    }

    int sgn(float x) {
        if (x > 0) return 1;
        if (x < 0) return -1;
        return 0;
    }

    std::vector<_Tetra> make_simplex(_Triangle face, std::vector<glm::vec3> point_set,
                                     std::vector<glm::vec3> gp, std::unordered_set<int> convex_hull) {
        auto& a        = gp[face.a];
        auto& b        = gp[face.b];
        auto& c        = gp[face.c];
        auto& opposite = gp[face.opposite];

        if (convex_hull.find(face.a) != convex_hull.end() &&//
            convex_hull.find(face.b) != convex_hull.end() &&//
            convex_hull.find(face.c) != convex_hull.end())
            return {};

        glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
        Plane halfplane(normal, (a + b + c) / 3.0f);

        std::vector<std::pair<float, _Tetra>> tetras;
        for (int i = 0; i < point_set.size(); i++) {
            int sgn_new = sgn(glm::dot(point_set[i] - halfplane.position, halfplane.normal));
            int sgn_opp = sgn(glm::dot(opposite - halfplane.position, halfplane.normal));

            if (point_set[i] == a || point_set[i] == b || point_set[i] == c || point_set[i] == opposite ||
                sgn_new == sgn_opp || sgn_new == 0) {
                continue;
            }

            float v =
              std::abs(glm::determinant(glm::mat4(a.x, a.y, a.z, 1.0f,                                //
                                                  b.x, b.y, b.z, 1.0f,                                //
                                                  c.x, c.y, c.z, 1.0f,                                //
                                                  point_set[i].x, point_set[i].y, point_set[i].z, 1.0f//
                                                  )) /
                       6.0f);
            if (v <= 0.1) continue;

            auto center  = get_circum_sphere_center(a, b, c, point_set[i]);
            float radius = glm::length(center - a);

            if (glm::dot(center - halfplane.position, halfplane.normal) >= 0.0f) { radius *= -1; }

            auto it = std::find(gp.begin(), gp.end(), point_set[i]);
            _Tetra tetra(face.a, face.b, face.c, std::distance(gp.begin(), it));

            tetras.push_back({radius, tetra});
        }

        std::sort(tetras.begin(), tetras.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<_Tetra> ret;
        std::transform(tetras.begin(), tetras.end(), std::back_inserter(ret),
                       [](const auto& e) { return e.second; });
        return ret;
    }

    void update(_Triangle face, std::vector<_Triangle>& afl) {
        auto it = std::find(afl.begin(), afl.end(), face);
        if (it != afl.end())
            afl.erase(it);
        else
            afl.push_back(face);
    }

    void add_face_to_afls(_Triangle face, Plane wall, std::vector<glm::vec3> p, std::vector<glm::vec3> p1,
                          std::vector<_Triangle>& afla, std::vector<_Triangle>& afl1,
                          std::vector<_Triangle>& afl2) {
        auto& a = p[face.a];
        auto& b = p[face.b];
        auto& c = p[face.c];
        if (wall.intersect(a, b - a) || wall.intersect(a, c - a) || wall.intersect(b, c - b))
            update(face, afla);
        else if (std::any_of(p1.begin(), p1.end(), [&a](const auto& e) { return e == a; }))
            update(face, afl1);
        else
            update(face, afl2);
    }

    void dewall(const std::vector<glm::vec3>& global_point_set, const std::unordered_set<int>& convex_hull,
                const std::vector<glm::vec3>& point_set, const std::vector<_Triangle>& afl,
                const glm::vec3& min, const glm::vec3& max, std::vector<_Tetra>& tetras) {
        std::vector<_Triangle> aflw, afl1, afl2;
        std::vector<glm::vec3> p1, p2;

        Plane wall(point_set, min, max);

        partition_point_set(point_set, wall, p1, p2);

        if (afl.empty()) {
            _Tetra t = make_first_simplex(point_set, wall);
            tetras.push_back(t);

            add_face_to_afls(t.abc(), wall, global_point_set, p1, aflw, afl1, afl2);
            add_face_to_afls(t.bcd(), wall, global_point_set, p1, aflw, afl1, afl2);
            add_face_to_afls(t.cda(), wall, global_point_set, p1, aflw, afl1, afl2);
            add_face_to_afls(t.dab(), wall, global_point_set, p1, aflw, afl1, afl2);
        } else {
            for (const _Triangle& face : afl) {
                add_face_to_afls(face, wall, global_point_set, p1, aflw, afl1, afl2);
            }
        }

        while (!aflw.empty()) {
            std::cout << tetras.size() << " - " << aflw.size() << std::endl;
            if (tetras.size() >= 1000) { return; }

            _Triangle face = aflw.back();

            std::vector<_Tetra> new_tetras = make_simplex(face, point_set, global_point_set, convex_hull);

            bool found = false;
            for (const auto& tetra : new_tetras) {
                if (std::none_of(tetras.begin(), tetras.end(),
                                 [&tetra            = std::as_const(tetra),
                                  &global_point_set = std::as_const(global_point_set)](const auto& e) {
                                     return e == tetra;
                                 })) {
                    tetras.push_back(tetra);

                    add_face_to_afls(tetra.abc(), wall, global_point_set, p1, aflw, afl1, afl2);
                    add_face_to_afls(tetra.bcd(), wall, global_point_set, p1, aflw, afl1, afl2);
                    add_face_to_afls(tetra.cda(), wall, global_point_set, p1, aflw, afl1, afl2);
                    add_face_to_afls(tetra.dab(), wall, global_point_set, p1, aflw, afl1, afl2);

                    found = true;
                    break;
                }
            }
            if (!found) aflw.pop_back();
        }

        if (!afl1.empty()) {
            auto p1_min = glm::vec3(min);
            auto p1_max = glm::vec3(max);

            if (!p1.empty()) {
                if (wall.normal.x > 0.0f) {
                    p1_max.x = wall.position.x;
                } else if (wall.normal.y > 0.0f) {
                    p1_max.y = wall.position.y;
                } else if (wall.normal.z > 0.0f) {
                    p1_max.z = wall.position.z;
                }
            }

            dewall(global_point_set, convex_hull, p1, afl1, p1_min, p1_max, tetras);
        }
        if (!afl2.empty()) {
            auto p2_min = glm::vec3(min);
            auto p2_max = glm::vec3(max);

            if (!p2.empty()) {
                if (wall.normal.x > 0.0f) {
                    p2_min.x = wall.position.x;
                } else if (wall.normal.y > 0.0f) {
                    p2_min.y = wall.position.y;
                } else if (wall.normal.z > 0.0f) {
                    p2_min.z = wall.position.z;
                }
            }

            dewall(global_point_set, convex_hull, p2, afl2, p2_min, p2_max, tetras);
        }
    }

    std::vector<_Tetra> dewall(const std::vector<glm::vec3>& point_set,
                               const std::unordered_set<int>& convex_hull) {
        glm::vec3 min(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max());
        glm::vec3 max(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest());

        for (const auto& point : convex_hull) {
            if (point_set[point].x < min.x)
                min.x = point_set[point].x;
            else if (point_set[point].x > max.x)
                max.x = point_set[point].x;

            if (point_set[point].y < min.y)
                min.y = point_set[point].y;
            else if (point_set[point].y > max.y)
                max.y = point_set[point].y;

            if (point_set[point].z < min.z)
                min.z = point_set[point].z;
            else if (point_set[point].z > max.z)
                max.z = point_set[point].z;
        }

        std::vector<_Tetra> tetras;
        dewall(point_set, convex_hull, point_set, {}, min, max, tetras);
        return tetras;
    }
}
