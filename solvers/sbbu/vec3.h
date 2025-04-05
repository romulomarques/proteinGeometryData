#pragma once
#include <cmath>

inline void vec3_set(double *x, double x0, double x1, double x2)
{
    x[0] = x0;
    x[1] = x1;
    x[2] = x2;
}

inline double vec3_norm(const double *x)
{
    return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

inline void vec3_unit(double *x)
{
    double nrm_x = vec3_norm(x);
    x[0] /= nrm_x;
    x[1] /= nrm_x;
    x[2] /= nrm_x;
}

// u = a * x + y
inline void vec3_axpy(double *u, double a, const double *x, const double *y)
{
    u[0] = a * x[0] + y[0];
    u[1] = a * x[1] + y[1];
    u[2] = a * x[2] + y[2];
}

// u = a * x + b * y
inline void vec3_axpby(double *u, double a, const double *x, double b, const double *y)
{
    u[0] = a * x[0] + b * y[0];
    u[1] = a * x[1] + b * y[1];
    u[2] = a * x[2] + b * y[2];
}

inline double vec3_dist2(const double *x, const double *y)
{
    const double a = x[0] - y[0];
    const double b = x[1] - y[1];
    const double c = x[2] - y[2];
    return a * a + b * b + c * c;
}

inline double vec3_dist(const double *x, const double *y)
{
    return sqrt(vec3_dist2(x, y));
}


inline double vec3_dot(const double *x, const double *y)
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

inline void vec3_cross(double *u, const double *x, const double *y)
{
    u[0] = x[1] * y[2] - x[2] * y[1];
    u[1] = x[2] * y[0] - x[0] * y[2];
    u[2] = x[0] * y[1] - x[1] * y[0];
}

inline void vec3_copy(double *u, const double *x)
{
    vec3_set(u, x[0], x[1], x[2]);
}