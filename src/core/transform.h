#pragma once

#include <geometry.h>
#include <cutil.h>

class SurfaceInteraction;

struct Matrix4x4 {
    Float m[4][4];
    
    __bidevice__ Matrix4x4() {
        m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;
        m[0][1] = m[0][2] = m[0][3] = m[1][0] = m[1][2] = m[1][3] = m[2][0] =
            m[2][1] = m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
    }
    
    __bidevice__ Matrix4x4(Float mat[4][4]);
    __bidevice__ Matrix4x4(Float t00, Float t01, Float t02, Float t03, Float t10, Float t11,
                           Float t12, Float t13, Float t20, Float t21, Float t22, Float t23,
                           Float t30, Float t31, Float t32, Float t33);
    
    __bidevice__ friend Matrix4x4 Transpose(const Matrix4x4 &);
    
    __bidevice__ static Matrix4x4 Mul(const Matrix4x4 &m1, const Matrix4x4 &m2) {
        Matrix4x4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            r.m[i][j] = m1.m[i][0] * m2.m[0][j] + m1.m[i][1] * m2.m[1][j] +
            m1.m[i][2] * m2.m[2][j] + m1.m[i][3] * m2.m[3][j];
        return r;
    }
    
    __bidevice__ friend Matrix4x4 Inverse(const Matrix4x4 &);
};

class Transform {
    public:
    Matrix4x4 m, mInv;
    // Transform Public Methods
    __bidevice__ Transform() {}
    __bidevice__ Transform(const Float mat[4][4]){
        m = Matrix4x4(mat[0][0], mat[0][1], mat[0][2], mat[0][3], mat[1][0],
                      mat[1][1], mat[1][2], mat[1][3], mat[2][0], mat[2][1],
                      mat[2][2], mat[2][3], mat[3][0], mat[3][1], mat[3][2],
                      mat[3][3]);
        mInv = Inverse(m);
    }
    
    __bidevice__ Transform(const Matrix4x4 &m) : m(m), mInv(Inverse(m)) {}
    __bidevice__ Transform(const Matrix4x4 &m, const Matrix4x4 &mInv) : m(m), mInv(mInv) {}
    
    __bidevice__ friend Transform Inverse(const Transform &t) {
        return Transform(t.mInv, t.m);
    }
    
    __bidevice__ friend Transform Transpose(const Transform &t) {
        return Transform(Transpose(t.m), Transpose(t.mInv));
    }
    
    __bidevice__ bool IsIdentity() const {
        return (m.m[0][0] == 1.f && m.m[0][1] == 0.f && m.m[0][2] == 0.f &&
                m.m[0][3] == 0.f && m.m[1][0] == 0.f && m.m[1][1] == 1.f &&
                m.m[1][2] == 0.f && m.m[1][3] == 0.f && m.m[2][0] == 0.f &&
                m.m[2][1] == 0.f && m.m[2][2] == 1.f && m.m[2][3] == 0.f &&
                m.m[3][0] == 0.f && m.m[3][1] == 0.f && m.m[3][2] == 0.f &&
                m.m[3][3] == 1.f);
    }
    
    __bidevice__ const Matrix4x4 &GetMatrix() const { return m; }
    __bidevice__ const Matrix4x4 &GetInverseMatrix() const { return mInv; }
    __bidevice__ bool HasScale() const {
        Float la2 = (*this)(vec3f(1, 0, 0)).LengthSquared();
        Float lb2 = (*this)(vec3f(0, 1, 0)).LengthSquared();
        Float lc2 = (*this)(vec3f(0, 0, 1)).LengthSquared();
#define NOT_ONE(x) ((x) < .999f || (x) > 1.001f)
        return (NOT_ONE(la2) || NOT_ONE(lb2) || NOT_ONE(lc2));
#undef NOT_ONE
    }
    
    template <typename T> inline __bidevice__ Point3<T> operator()(const Point3<T> &p) const;
    template <typename T> inline __bidevice__ vec3<T> operator()(const vec3<T> &v) const;
    template <typename T> inline __bidevice__ Normal3<T> operator()(const Normal3<T> &) const;
    
    inline __bidevice__ Ray operator()(const Ray &r) const;
    inline __bidevice__ RayDifferential operator()(const RayDifferential &r) const;
    
    __bidevice__ Transform operator*(const Transform &t2) const;
    __bidevice__ bool SwapsHandedness() const;
    
    //Bounds3f operator()(const Bounds3f &b) const;
    __bidevice__ SurfaceInteraction operator()(const SurfaceInteraction &si) const;
    
    template <typename T> inline __bidevice__ Point3<T> operator()(const Point3<T> &pt,
                                                                   vec3<T> *absError) const;
    template <typename T> inline __bidevice__ Point3<T> operator()(const Point3<T> &p, const vec3<T> &pError,
                                                                   vec3<T> *pTransError) const;
    template <typename T> inline __bidevice__ vec3<T> operator()(const vec3<T> &v,
                                                                 vec3<T> *vTransError) const;
    template <typename T> inline __bidevice__ vec3<T> operator()(const vec3<T> &v, const vec3<T> &vError,
                                                                 vec3<T> *vTransError) const;
    inline __bidevice__ Ray operator()(const Ray &r, vec3f *oError,
                                       vec3f *dError) const;
    inline __bidevice__ Ray operator()(const Ray &r, const vec3f &oErrorIn,
                                       const vec3f &dErrorIn, vec3f *oErrorOut,
                                       vec3f *dErrorOut) const;
};

__bidevice__ Transform Translate(const vec3f &delta);
__bidevice__ Transform Scale(Float x, Float y, Float z);
__bidevice__ Transform RotateX(Float theta);
__bidevice__ Transform RotateY(Float theta);
__bidevice__ Transform RotateZ(Float theta);
__bidevice__ Transform Rotate(Float theta, const vec3f &axis);
__bidevice__ Transform LookAt(const Point3f &pos, const Point3f &look, const vec3f &up);
__bidevice__ Transform Orthographic(Float znear, Float zfar);
__bidevice__ Transform Perspective(Float fov, Float znear, Float zfar);
__bidevice__ bool SolveLinearSystem2x2(const Float A[2][2], const Float B[2], Float *x0,
                                       Float *x1);

// Transform Inline Functions
template <typename T> inline __bidevice__ 
Point3<T> Transform::operator()(const Point3<T> &p) const{
    T x = p.x, y = p.y, z = p.z;
    T xp = m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z + m.m[0][3];
    T yp = m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z + m.m[1][3];
    T zp = m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z + m.m[2][3];
    T wp = m.m[3][0] * x + m.m[3][1] * y + m.m[3][2] * z + m.m[3][3];
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T> inline __bidevice__ 
vec3<T> Transform::operator()(const vec3<T> &v) const{
    T x = v.x, y = v.y, z = v.z;
    return vec3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                   m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                   m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
}

template <typename T> inline __bidevice__  
Normal3<T> Transform::operator()(const Normal3<T> &n) const{
    T x = n.x, y = n.y, z = n.z;
    return Normal3<T>(mInv.m[0][0] * x + mInv.m[1][0] * y + mInv.m[2][0] * z,
                      mInv.m[0][1] * x + mInv.m[1][1] * y + mInv.m[2][1] * z,
                      mInv.m[0][2] * x + mInv.m[1][2] * y + mInv.m[2][2] * z);
}

inline __bidevice__ Ray Transform::operator()(const Ray &r) const{
    vec3f oError;
    Point3f o = (*this)(r.o, &oError);
    vec3f d = (*this)(r.d);
    
    Float lengthSquared = d.LengthSquared();
    Float tMax = r.tMax;
    if (lengthSquared > 0) {
        Float dt = Dot(Abs(d), oError) / lengthSquared;
        o += d * dt;
        tMax -= dt;
    }
    return Ray(o, d, tMax, r.time);
}

inline __bidevice__  RayDifferential Transform::operator()(const RayDifferential &r) const{
    Ray tr = (*this)(Ray(r));
    RayDifferential ret(tr.o, tr.d, tr.tMax, tr.time);
    ret.hasDifferentials = r.hasDifferentials;
    ret.rxOrigin = (*this)(r.rxOrigin);
    ret.ryOrigin = (*this)(r.ryOrigin);
    ret.rxDirection = (*this)(r.rxDirection);
    ret.ryDirection = (*this)(r.ryDirection);
    return ret;
}

template <typename T> inline __bidevice__ 
Point3<T> Transform::operator()(const Point3<T> &p, vec3<T> *pError) const{
    T x = p.x, y = p.y, z = p.z;
    T xp = (m.m[0][0] * x + m.m[0][1] * y) + (m.m[0][2] * z + m.m[0][3]);
    T yp = (m.m[1][0] * x + m.m[1][1] * y) + (m.m[1][2] * z + m.m[1][3]);
    T zp = (m.m[2][0] * x + m.m[2][1] * y) + (m.m[2][2] * z + m.m[2][3]);
    T wp = (m.m[3][0] * x + m.m[3][1] * y) + (m.m[3][2] * z + m.m[3][3]);
    
    T xAbsSum = (Absf(m.m[0][0] * x) + Absf(m.m[0][1] * y) +
                 Absf(m.m[0][2] * z) + Absf(m.m[0][3]));
    T yAbsSum = (Absf(m.m[1][0] * x) + Absf(m.m[1][1] * y) +
                 Absf(m.m[1][2] * z) + Absf(m.m[1][3]));
    T zAbsSum = (Absf(m.m[2][0] * x) + Absf(m.m[2][1] * y) +
                 Absf(m.m[2][2] * z) + Absf(m.m[2][3]));
    *pError = gamma(3) * vec3<T>(xAbsSum, yAbsSum, zAbsSum);
    Assert(wp != 0);
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T> inline __bidevice__ 
Point3<T> Transform::operator()(const Point3<T> &pt, const vec3<T> &ptError,
                                vec3<T> *absError) const
{
    T x = pt.x, y = pt.y, z = pt.z;
    T xp = (m.m[0][0] * x + m.m[0][1] * y) + (m.m[0][2] * z + m.m[0][3]);
    T yp = (m.m[1][0] * x + m.m[1][1] * y) + (m.m[1][2] * z + m.m[1][3]);
    T zp = (m.m[2][0] * x + m.m[2][1] * y) + (m.m[2][2] * z + m.m[2][3]);
    T wp = (m.m[3][0] * x + m.m[3][1] * y) + (m.m[3][2] * z + m.m[3][3]);
    absError->x =
        (gamma(3) + (T)1) *
        (Absf(m.m[0][0]) * ptError.x + Absf(m.m[0][1]) * ptError.y +
         Absf(m.m[0][2]) * ptError.z) +
        gamma(3) * (Absf(m.m[0][0] * x) + Absf(m.m[0][1] * y) +
                    Absf(m.m[0][2] * z) + Absf(m.m[0][3]));
    absError->y =
        (gamma(3) + (T)1) *
        (Absf(m.m[1][0]) * ptError.x + Absf(m.m[1][1]) * ptError.y +
         Absf(m.m[1][2]) * ptError.z) +
        gamma(3) * (Absf(m.m[1][0] * x) + Absf(m.m[1][1] * y) +
                    Absf(m.m[1][2] * z) + Absf(m.m[1][3]));
    absError->z =
        (gamma(3) + (T)1) *
        (Absf(m.m[2][0]) * ptError.x + Absf(m.m[2][1]) * ptError.y +
         Absf(m.m[2][2]) * ptError.z) +
        gamma(3) * (Absf(m.m[2][0] * x) + Absf(m.m[2][1] * y) +
                    Absf(m.m[2][2] * z) + Absf(m.m[2][3]));
    Assert(wp != 0);
    if (wp == 1.)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T> inline __bidevice__ 
vec3<T> Transform::operator()(const vec3<T> &v, vec3<T> *absError) const{
    T x = v.x, y = v.y, z = v.z;
    absError->x =
        gamma(3) * (Absf(m.m[0][0] * v.x) + Absf(m.m[0][1] * v.y) +
                    Absf(m.m[0][2] * v.z));
    absError->y =
        gamma(3) * (Absf(m.m[1][0] * v.x) + Absf(m.m[1][1] * v.y) +
                    Absf(m.m[1][2] * v.z));
    absError->z =
        gamma(3) * (Absf(m.m[2][0] * v.x) + Absf(m.m[2][1] * v.y) +
                    Absf(m.m[2][2] * v.z));
    return vec3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                   m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                   m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
}

template <typename T> inline __bidevice__  
vec3<T> Transform::operator()(const vec3<T> &v,const vec3<T> &vError,
                              vec3<T> *absError) const
{
    T x = v.x, y = v.y, z = v.z;
    absError->x =
        (gamma(3) + (T)1) *
        (Absf(m.m[0][0]) * vError.x + Absf(m.m[0][1]) * vError.y +
         Absf(m.m[0][2]) * vError.z) +
        gamma(3) * (Absf(m.m[0][0] * v.x) + Absf(m.m[0][1] * v.y) +
                    Absf(m.m[0][2] * v.z));
    absError->y =
        (gamma(3) + (T)1) *
        (Absf(m.m[1][0]) * vError.x + Absf(m.m[1][1]) * vError.y +
         Absf(m.m[1][2]) * vError.z) +
        gamma(3) * (Absf(m.m[1][0] * v.x) + Absf(m.m[1][1] * v.y) +
                    Absf(m.m[1][2] * v.z));
    absError->z =
        (gamma(3) + (T)1) *
        (Absf(m.m[2][0]) * vError.x + Absf(m.m[2][1]) * vError.y +
         Absf(m.m[2][2]) * vError.z) +
        gamma(3) * (Absf(m.m[2][0] * v.x) + Absf(m.m[2][1] * v.y) +
                    Absf(m.m[2][2] * v.z));
    return vec3<T>(m.m[0][0] * x + m.m[0][1] * y + m.m[0][2] * z,
                   m.m[1][0] * x + m.m[1][1] * y + m.m[1][2] * z,
                   m.m[2][0] * x + m.m[2][1] * y + m.m[2][2] * z);
}

inline __bidevice__ Ray Transform::operator()(const Ray &r, vec3f *oError,
                                              vec3f *dError) const
{
    Point3f o = (*this)(r.o, oError);
    vec3f d = (*this)(r.d, dError);
    Float tMax = r.tMax;
    Float lengthSquared = d.LengthSquared();
    if (lengthSquared > 0) {
        Float dt = Dot(Abs(d), *oError) / lengthSquared;
        o += d * dt;
    }
    return Ray(o, d, tMax, r.time);
}

inline __bidevice__ Ray Transform::operator()(const Ray &r, const vec3f &oErrorIn,
                                              const vec3f &dErrorIn, vec3f *oErrorOut,
                                              vec3f *dErrorOut) const
{
    Point3f o = (*this)(r.o, oErrorIn, oErrorOut);
    vec3f d = (*this)(r.d, dErrorIn, dErrorOut);
    Float tMax = r.tMax;
    Float lengthSquared = d.LengthSquared();
    if (lengthSquared > 0) {
        Float dt = Dot(Abs(d), *oErrorOut) / lengthSquared;
        o += d * dt;
    }
    return Ray(o, d, tMax, r.time);
}