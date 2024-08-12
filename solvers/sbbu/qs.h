#pragma once
#include <cstdlib>
#include <stdexcept>
#include "vec3.h"

double mat3x3_det(double A[3][3], int j, double *v);
void mat3x3_cramer(double *x, double A[3][3], double *y);
void qs_p(double *P, const double *N, const double *A, double ax2, const double *B, double bx2, const double *C, double cx2, double dtol);

// Replaces j-th column by v
double mat3x3_det(double A[3][3], int j, double *v)
{

    if (v == NULL)
        return A[0][0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
               A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0] -
               A[0][1] * A[1][0] * A[2][2] - A[0][0] * A[1][2] * A[2][1];

    if (j == 0)
        return v[0] * A[1][1] * A[2][2] + A[0][1] * A[1][2] * v[2] +
               A[0][2] * v[1] * A[2][1] - A[0][2] * A[1][1] * v[2] -
               A[0][1] * v[1] * A[2][2] - v[0] * A[1][2] * A[2][1];

    if (j == 1)
        return A[0][0] * v[1] * A[2][2] + v[0] * A[1][2] * A[2][0] +
               A[0][2] * A[1][0] * v[2] - A[0][2] * v[1] * A[2][0] -
               v[0] * A[1][0] * A[2][2] - A[0][0] * A[1][2] * v[2];

    if (j == 2)
        return A[0][0] * A[1][1] * v[2] + A[0][1] * v[1] * A[2][0] +
               v[0] * A[1][0] * A[2][1] - v[0] * A[1][1] * A[2][0] -
               A[0][1] * A[1][0] * v[2] - A[0][0] * v[1] * A[2][1];

    throw std::runtime_error("P is not the solution of M*P=y");
}

void mat3x3_cramer(double *x, double A[3][3], double *y)
{
    double dA = mat3x3_det(A, -1, NULL);
    double d0 = mat3x3_det(A, 0, y);
    double d1 = mat3x3_det(A, 1, y);
    double d2 = mat3x3_det(A, 2, y);
    x[0] = d0 / dA;
    x[1] = d1 / dA;
    x[2] = d2 / dA;
}

void qs_p(double *P, const double *N, const double *A, double ax2, const double *B, double bx2, const double *C, double cx2, double dtol)
{
    double A2 = vec3_norm(A);
    double B2 = vec3_norm(B);
    double C2 = vec3_norm(C);
    A2 *= A2;
    B2 *= B2;
    C2 *= C2;

    // set linear system
    double M[][3] = {{N[0], N[1], N[2]}, {A[0] - B[0], A[1] - B[1], A[2] - B[2]}, {A[0] - C[0], A[1] - C[1], A[2] - C[2]}};
    double y[] = {vec3_dot(A, N), (bx2 - ax2 + A2 - B2) / 2.0, (cx2 - ax2 + A2 - C2) / 2.0};

    // solve the system M*P=y
    mat3x3_cramer(P, M, y);

#ifdef DEBUG
    for (int k = 0; k < 3; ++k)
        if (fabs(vec3_dot(&M[k][0], P) - y[k]) > dtol)
            throw std::runtime_error("P is not the solution of M*P=y");
#endif
}
