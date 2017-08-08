//
// Created by user on 8/8/17.
//

#ifndef RECONSTRUCTION_REFINE_MESH_H
#define RECONSTRUCTION_REFINE_MESH_H

#include <tuple>
#include <math.h>
#include <vector>

struct vec3f {
    double x, y, z;

    inline vec3f(void) {}

    inline vec3f(const double X, const double Y, const double Z) {
        x = X; y = Y; z = Z;
    }

    inline vec3f operator+(vec3f const & a) const {
        return vec3f(x + a.x, y + a.y, z + a.z);
    }

    inline vec3f operator+=(vec3f const & a) const {
        return vec3f( x + a.x, y + a.y, z + a.z );
    }

    inline vec3f operator*(double const a) const {
        return vec3f( x * a, y * a, z * a );
    }

    inline vec3f operator*(vec3f const & a) const {
        return vec3f( x * a.x, y * a.y, z * a.z );
    }

    inline vec3f operator=(vec3f const & a) {
        x = a.x;
        y = a.y;
        z = a.z;
        return *this;
    }

    inline vec3f operator/(vec3f const & a) const {
        return vec3f(x / a.x, y / a.y, z / a.z);
    }

    inline vec3f operator-(vec3f const & a) const {
        return vec3f(x - a.x, y - a.y, z - a.z );
    }

    inline vec3f operator/(double const a) const {
        return vec3f(x / a, y / a, z / a);
    }

    inline double dot(vec3f const & a ) const {
        return a.x * x + a.y * y + a.z * z;
    }

    inline vec3f cross(vec3f const & a, vec3f const & b) {
        x = a.y * b.z - a.z * b.y;
        y = a.z * b.x - a.x * b.z;
        z = a.x * b.y - a.y * b.x;
        return *this;
    }

    inline vec3f normalize(double desired_length = 1) {
        double square = sqrt(x * x + y * y + z * z);
        x /= square;
        y /= square;
        z /= square;
        return *this;
    }
};

class SymmetricMatrix {
    double m[10];
public:
    // Constructor
    SymmetricMatrix(double c = 0) {
        for (int i = 0; i < 10; ++i) {
            m[i] = c;
        }
    }

    SymmetricMatrix(double m11, double m12, double m13, double m14,
                    double m22, double m23, double m24,
                    double m33, double m34,
                    double m44)
    {
        m[0] = m11; m[1] = m12; m[2] = m13; m[3] = m14;
        m[4] = m22; m[5] = m23; m[6] = m24;
        m[7] = m33; m[8] = m34;
        m[9] = m44;
    }

    // Make plane
    SymmetricMatrix(double a, double b, double c, double d) {
        m[0] = a * a;  m[1] = a * b;  m[2] = a * c;  m[3] = a * d;
        m[4] = b * b;  m[5] = b * c;  m[6] = b * d;
        m[7] = c * c;  m[8] = c * d;
        m[9] = d * d;
    }

    double operator[](int c) const {
        return m[c];
    }

    // Determinant

    double det(int a11, int a12, int a13,
               int a21, int a22, int a23,
               int a31, int a32, int a33)
    {
        double det =  m[a11]*m[a22]*m[a33] + m[a13]*m[a21]*m[a32] + m[a12]*m[a23]*m[a31]
                      - m[a13]*m[a22]*m[a31] - m[a11]*m[a23]*m[a32] - m[a12]*m[a21]*m[a33];
        return det;
    }

    const SymmetricMatrix operator+(SymmetricMatrix const & n) const	{
        return SymmetricMatrix(m[0] + n[0], m[1] + n[1], m[2] + n[2], m[3] + n[3],
                               m[4] + n[4], m[5] + n[5], m[6] + n[6],
                               m[7] + n[7], m[8] + n[8],
                               m[9] + n[9]);
    }

    SymmetricMatrix& operator+=(SymmetricMatrix const & n) {
        m[0] += n[0]; m[1] += n[1]; m[2] += n[2]; m[3] += n[3];
        m[4] += n[4]; m[5] += n[5]; m[6] += n[6];
        m[7] += n[7]; m[8] += n[8];
        m[9] += n[9];
        return *this;
    }
};

typedef unsigned long ulong;

class MeshSimplify {
public:
    struct Triangle {
        ulong v[3] = {0, 0, 0};
        double err[4] = {0.0, 0.0, 0.0, 0.0};
        int deleted = 0, dirty = 0;
        vec3f n = vec3f(0, 0, 0);

        // Constructor and method
        Triangle() {};

        void update(ulong const x, ulong const y, ulong const z) {
            v[0] = x;
            v[1] = y;
            v[2] = z;
        }

        std::tuple<ulong, ulong ,ulong> get_coord() const {
            return std::make_tuple(v[0], v[1], v[2]);
        };
    };

    struct Vertex {
        vec3f p = vec3f(0, 0, 0);
        ulong tstart = 0, tcount = 0;
        SymmetricMatrix q;
        int border = 0;

        // Constructor and method
        Vertex() {};

        void update(double const x, double const y, double const z) {
            p.x = x;
            p.y = y;
            p.z = z;
        }

        std::tuple<double, double, double> get_coord() const {
            return std::make_tuple(p.x, p.y, p.z);
        };
    };
    //
    // Main simplification function
    //
    // target_count  : target nr. of triangles
    // agressiveness : sharpness to increase the threshold.
    //                 5..8 are good numbers
    //                 more iterations yield higher quality
    void simplify_mesh(ulong const target_count, double const agressiveness, bool const verbose);

    MeshSimplify (ulong const v_count, ulong const t_count) {
        vertices.reserve(v_count);
        triangles.reserve(t_count);
    }
    std::vector<Triangle> triangles;
    std::vector<Vertex> vertices;
private:
    struct Ref {
        int tid, tvertex;
    };
    std::vector<Ref> refs;

    // Helper functions
    double vertex_error(SymmetricMatrix const &q, double const x, double const y, double const z);

    double calculate_error(int id_v1, int id_v2, vec3f &p_result);

    bool flipped(vec3f const &p, int const i1, Vertex const &v0, Vertex const &v1, std::vector<int> &deleted);

    void update_triangles(int const i0, Vertex const &v, std::vector<int> const &deleted, int &deleted_triangles);

    void update_mesh(int const iteration);

    void compact_mesh();
};

#endif //RECONSTRUCTION_REFINE_MESH_H
