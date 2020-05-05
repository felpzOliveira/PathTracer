#if !defined(TRANSFORM_H)
#define TRANSFORM_H
#include <cutil.h>
#include <types.h>
#include <utilities.h>

struct Matrix4x4 {
    __bidevice__ Matrix4x4() {
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
        m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
            m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
    }
    
    __bidevice__ 
        Matrix4x4(float mat[4][4]);
    
    __bidevice__ 
        Matrix4x4(float t00, float t01, float t02, float t03, float t10, float t11,
                  float t12, float t13, float t20, float t21, float t22, float t23,
                  float t30, float t31, float t32, float t33);
    
    __bidevice__ bool operator==(const Matrix4x4 &m2) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            if (m[i][j] != m2.m[i][j]) return false;
        return true;
    }
    
    __bidevice__ bool operator!=(const Matrix4x4 &m2) const {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            if (m[i][j] != m2.m[i][j]) return true;
        return false;
    }
    
    __bidevice__ friend  Matrix4x4 Transpose(const Matrix4x4 &);
    
    __bidevice__  static  Matrix4x4 Mul(const Matrix4x4 &m1,
                                        const Matrix4x4 &m2) 
    {
        Matrix4x4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
            m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
        return r;
    }
    
    __bidevice__ friend  Matrix4x4 Inverse(const Matrix4x4 &);
    
    float m[4][4];
};

__bidevice__ 
inline bool SolveLinearSystem2x2(const float A[2][2], const float B[2], float *x0,
                                 float *x1) {
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (ABS(det) < 1e-10f) return false;
    *x0 = (A[1][1] * B[0] - A[0][1] * B[1]) / det;
    *x1 = (A[0][0] * B[1] - A[1][0] * B[0]) / det;
    if (isnan(*x0) || isnan(*x1)) return false;
    return true;
}

__bidevice__ 
inline Matrix4x4::Matrix4x4(float mat[4][4]) { memcpy(m, mat, 16 * sizeof(float)); }

__bidevice__ 
inline Matrix4x4::Matrix4x4(float t00, float t01, float t02, float t03, float t10,
                            float t11, float t12, float t13, float t20, float t21,
                            float t22, float t23, float t30, float t31, float t32,
                            float t33) {
    m[0][0] = t00;
    m[0][1] = t01;
    m[0][2] = t02;
    m[0][3] = t03;
    m[1][0] = t10;
    m[1][1] = t11;
    m[1][2] = t12;
    m[1][3] = t13;
    m[2][0] = t20;
    m[2][1] = t21;
    m[2][2] = t22;
    m[2][3] = t23;
    m[3][0] = t30;
    m[3][1] = t31;
    m[3][2] = t32;
    m[3][3] = t33;
}

__bidevice__ inline Matrix4x4 Transpose(const Matrix4x4 &m) {
    return Matrix4x4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0], m.m[0][1],
                     m.m[1][1], m.m[2][1], m.m[3][1], m.m[0][2], m.m[1][2],
                     m.m[2][2], m.m[3][2], m.m[0][3], m.m[1][3], m.m[2][3],
                     m.m[3][3]);
}

__bidevice__ inline Matrix4x4 Inverse(const Matrix4x4 &m) {
    int indxc[4], indxr[4];
    int ipiv[4] = {0, 0, 0, 0};
    float minv[4][4];
    memcpy(minv, m.m, 4 * 4 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        int irow = 0, icol = 0;
        float big = 0.f;
        // Choose pivot
        for (int j = 0; j < 4; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < 4; k++) {
                    if (ipiv[k] == 0) {
                        if (ABS(minv[j][k]) >= big) {
                            big = float(ABS(minv[j][k]));
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ++ipiv[icol];
        // Swap rows _irow_ and _icol_ for pivot
        if (irow != icol) {
            for (int k = 0; k < 4; ++k) {
                float aux = minv[irow][k];
                minv[irow][k] = minv[icol][k];
                minv[icol][k] = aux;
            }
        }
        indxr[i] = irow;
        indxc[i] = icol;
        
        // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
        float pivinv = 1. / minv[icol][icol];
        minv[icol][icol] = 1.;
        for (int j = 0; j < 4; j++) minv[icol][j] *= pivinv;
        
        // Subtract this row from others to zero out their columns
        for (int j = 0; j < 4; j++) {
            if (j != icol) {
                float save = minv[j][icol];
                minv[j][icol] = 0;
                for (int k = 0; k < 4; k++) minv[j][k] -= minv[icol][k] * save;
            }
        }
    }
    // Swap columns to reflect permutation
    for (int j = 3; j >= 0; j--) {
        if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++){
                float aux = minv[k][indxr[j]];
                minv[k][indxr[j]] = minv[k][indxc[j]];
                minv[k][indxc[j]] = aux;
            }
        }
    }
    return Matrix4x4(minv);
}

class Transform{
    public:
    Matrix4x4 m, invM;
    __bidevice__ Transform(){}
    __bidevice__ Transform(const Transform &o){
        m = o.m;
        invM = o.invM;
    }
    
    __bidevice__ Transform(Matrix4x4 _m){
        m = _m;
        invM = Inverse(m);
    }
    
    __bidevice__ Transform(Matrix4x4 _m, Matrix4x4 _im){
        m = _m;
        invM = _im;
    }
    
    __bidevice__ friend Transform * Inverse(Transform *t){
        return new Transform(t->invM, t->m);
    }
    
    __bidevice__ friend Transform Inverse(const Transform &t){
        return Transform(t.invM, t.m);
    }
    
    __bidevice__ friend Transform Transpose(const Transform &t){
        return Transform(Transpose(t.m), Transpose(t.invM));
    }
    
    __bidevice__ Transform operator*(const Transform &t2) const{
        return Transform(Matrix4x4::Mul(m, t2.m),
                         Matrix4x4::Mul(t2.invM, invM));
    }
    
    __bidevice__ glm::vec3 point(glm::vec3 p){
        float x = p.x, y = p.y, z = p.z;
        float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
        float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
        float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
        float wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
        float inv = 1.0f / wp;
        return glm::vec3(xp * inv, yp * inv, zp * inv);
    }
    
    __bidevice__ bool SwapsHandeness(){
        float det = (m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) -
                     m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0]) +
                     m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]));
        return det < 0;
    }
    
    __bidevice__ glm::vec3 point(glm::vec3 p, glm::vec3 *pError){
        float x = p.x, y = p.y, z = p.z;
        float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
        float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
        float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
        float wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
        
        float xAbs = ABS(m.m[0][0]*x) + ABS(m.m[0][1]*y) + ABS(m.m[0][2]*z) + ABS(m.m[0][3]);
        
        float yAbs = ABS(m.m[1][0]*x) + ABS(m.m[1][1]*y) + ABS(m.m[1][2]*z) + ABS(m.m[1][3]);
        
        float zAbs = ABS(m.m[2][0]*x) + ABS(m.m[2][1]*y) + ABS(m.m[2][2]*z) + ABS(m.m[2][3]);
        
        *pError = glm::vec3(xAbs, yAbs, zAbs) * (float)gamma(3);
        
        return glm::vec3(xp, yp, zp) / wp;
    }
    
    __bidevice__ glm::vec3 point(glm::vec3 pt, glm::vec3 ptError,
                                 glm::vec3 *absError)
    {
        float x = pt.x, y = pt.y, z = pt.z;
        float xp = (m.m[0][0] * x + m.m[0][1] * y) + (m.m[0][2] * z + m.m[0][3]);
        float yp = (m.m[1][0] * x + m.m[1][1] * y) + (m.m[1][2] * z + m.m[1][3]);
        float zp = (m.m[2][0] * x + m.m[2][1] * y) + (m.m[2][2] * z + m.m[2][3]);
        float wp = (m.m[3][0] * x + m.m[3][1] * y) + (m.m[3][2] * z + m.m[3][3]);
        absError->x =
            (gamma(3) + (float)1) *
            (ABS(m.m[0][0]) * ptError.x + ABS(m.m[0][1]) * ptError.y + ABS(m.m[0][2]) * ptError.z) +
            gamma(3) * (ABS(m.m[0][0] * x) + ABS(m.m[0][1] * y) + ABS(m.m[0][2] * z) + ABS(m.m[0][3]));
        
        absError->y =
            (gamma(3) + (float)1) *
            (ABS(m.m[1][0]) * ptError.x + ABS(m.m[1][1]) * ptError.y +
             ABS(m.m[1][2]) * ptError.z) +
            gamma(3) * (ABS(m.m[1][0] * x) + ABS(m.m[1][1] * y) +
                        ABS(m.m[1][2] * z) + ABS(m.m[1][3]));
        
        absError->z =
            (gamma(3) + (float)1) *
            (ABS(m.m[2][0]) * ptError.x + ABS(m.m[2][1]) * ptError.y +
             ABS(m.m[2][2]) * ptError.z) +
            gamma(3) * (ABS(m.m[2][0] * x) + ABS(m.m[2][1] * y) +
                        ABS(m.m[2][2] * z) + ABS(m.m[2][3]));
        if (wp == 1.)
            return glm::vec3(xp, yp, zp);
        else
            return glm::vec3(xp, yp, zp) / wp; 
        
    }
    
    __bidevice__ glm::vec3 vector(glm::vec3 v){
        float x = v.x, y = v.y, z = v.z;
        float xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z;
        float yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z;
        float zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z;
        return glm::vec3(xp, yp, zp);
    }
    
    __bidevice__ glm::vec3 normal(glm::vec3 n){
        float x = n.x, y = n.y, z = n.z;
        float xp = invM.m[0][0] * x + invM.m[1][0] * y + invM.m[2][0] * z;
        float yp = invM.m[0][1] * x + invM.m[1][1] * y + invM.m[2][1] * z;
        float zp = invM.m[0][2] * x + invM.m[1][2] * y + invM.m[2][2] * z;
        return glm::vec3(xp, yp, zp);
    }
};

inline __bidevice__ Transform Scale(float x, float y, float z){
    Matrix4x4 m(x, 0, 0, 0,
                0, y, 0, 0,
                0, 0, z, 0,
                0, 0, 0, 1);
    
    Matrix4x4 inv(1.0f/x, 0,     0,     0,
                  0,     1.0f/y, 0,     0,
                  0,      0,    1.0f/z, 0,
                  0,      0,     0,     1);
    return Transform(m, inv);
}

inline __bidevice__ Transform Scale(float s){
    return Scale(s, s, s);
}

inline __bidevice__ Transform Scale(glm::vec3 v){
    return Scale(v.x, v.y, v.z);
}

inline __bidevice__ Transform Translate(glm::vec3 delta){
    Matrix4x4 m(1, 0, 0, delta.x,
                0, 1, 0, delta.y,
                0, 0, 1, delta.z,
                0, 0, 0,    1);
    
    Matrix4x4 inv(1, 0, 0, -delta.x,
                  0, 1, 0, -delta.y,
                  0, 0, 1, -delta.z,
                  0, 0, 0,    1);
    return Transform(m, inv);
}

inline __bidevice__ Transform Rotate(float theta, glm::vec3 axis){
    glm::vec3 a = glm::normalize(axis);
    float sinTheta = glm::sin(glm::radians(theta));
    float cosTheta = glm::cos(glm::radians(theta));
    Matrix4x4 m;
    // Compute rotation of first basis vector
    m.m[0][0] = a.x * a.x + (1 - a.x * a.x) * cosTheta;
    m.m[0][1] = a.x * a.y * (1 - cosTheta) - a.z * sinTheta;
    m.m[0][2] = a.x * a.z * (1 - cosTheta) + a.y * sinTheta;
    m.m[0][3] = 0;
    
    // Compute rotations of second and third basis vectors
    m.m[1][0] = a.x * a.y * (1 - cosTheta) + a.z * sinTheta;
    m.m[1][1] = a.y * a.y + (1 - a.y * a.y) * cosTheta;
    m.m[1][2] = a.y * a.z * (1 - cosTheta) - a.x * sinTheta;
    m.m[1][3] = 0;
    
    m.m[2][0] = a.x * a.z * (1 - cosTheta) - a.y * sinTheta;
    m.m[2][1] = a.y * a.z * (1 - cosTheta) + a.x * sinTheta;
    m.m[2][2] = a.z * a.z + (1 - a.z * a.z) * cosTheta;
    m.m[2][3] = 0;
    return Transform(m, Transpose(m));
}

#endif