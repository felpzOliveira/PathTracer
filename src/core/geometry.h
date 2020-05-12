#pragma once
#include <math.h>
#include <cutil.h>
#include <cfloat>
#include <stdio.h>

#define Assert(x) __assert_check((x), #x, __FILE__, __LINE__)
#define Infinity FLT_MAX
#define __vec3_strfmtA(v) "%s = [%g %g %g]"
#define __vec3_strfmt(v) "[%g %g %g]"
#define __vec3_args(v) v.x, v.y, v.z
#define __vec3_argsA(v) #v, v.x, v.y, v.z

#define OneMinusEpsilon 0.99999994f
#define ShadowEpsilon 0.0001f
#define Pi 3.14159265358979323846
#define InvPi 0.31830988618379067154
#define Inv2Pi 0.15915494309189533577
#define Inv4Pi 0.07957747154594766788
#define PiOver2 1.57079632679489661923
#define PiOver4 0.78539816339744830961
#define Sqrt2 1.41421356237309504880
#define MachineEpsilon 5.96046e-08f

#define MIN_FLT -FLT_MAX

typedef float Float;
//typedef double Float;

enum TransportMode{
    Radiance = 0,
};

inline __bidevice__ void __assert_check(bool v, const char *name, 
                                        const char *filename, int line)
{
    if(!v){
        int* ptr = nullptr;
        printf("Assert: %s (%s:%d)\n", name, filename, line);
        *ptr = 10;
    }
}

inline __bidevice__ Float Max(Float a, Float b){ return a > b ? a : b; }
inline __bidevice__ Float Min(Float a, Float b){ return a < b ? a : b; }
inline __bidevice__ Float Absf(Float v){ return v > 0 ? v : -v; }
inline __bidevice__ bool IsNaN(Float v){ return v != v; }
inline __bidevice__ Float Radians(Float deg) { return (Pi / 180) * deg; }
inline __bidevice__ bool IsZero(Float a){ return Absf(a) < 0.00001; }

template <typename T, typename U, typename V> 
inline __bidevice__ T Clamp(T val, U low, V high){
    if(val < low) return low;
    if(val > high) return high;
    return val;
}

template<typename T>
inline __bidevice__ Float gamma(T n){ 
    return ((Float)n * MachineEpsilon) / (1 - (Float)n * MachineEpsilon); 
}

__bidevice__ inline void swap(Float *a, Float *b){
    Float aux = *a; *a = *b; *b = aux;
}

__bidevice__ inline void swap(Float &a, Float &b){
    Float aux = a; a = b; b = aux;
}

inline __bidevice__ bool Quadratic(Float a, Float b, Float c, Float *t0, Float *t1) {
    double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
    if(discrim < 0) return false;
    double rootDiscrim = std::sqrt(discrim);
    double q;
    if(b < 0)
        q = -.5 * (b - rootDiscrim);
    else
        q = -.5 * (b + rootDiscrim);
    *t0 = q / a;
    *t1 = c / q;
    if(*t0 > *t1) swap(t0, t1);
    return true;
}


template<typename T> class vec2{
    public:
    T x, y;
    
    __bidevice__ vec2(){ x = y = (T)0; }
    __bidevice__ vec2(T a){ x = y = a; }
    __bidevice__ vec2(T a, T b): x(a), y(b){
        Assert(!HasNaN());
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y);
    }
    
    __bidevice__ T operator[](int i) const{
        Assert(i >= 0 && i < 2);
        if(i == 0) return x;
        return y;
    }
    
    __bidevice__ T &operator[](int i){
        Assert(i >= 0 && i < 2);
        if(i == 0) return x;
        return y;
    }
    
    __bidevice__ vec2<T> operator/(T f) const{
        Assert(f != 0);
        Float inv = (Float)1 / f;
        return vec2<T>(x * inv, y * inv);
    }
    
    __bidevice__ vec2<T> &operator/(T f){
        Assert(f != 0);
        Float inv = (Float)1 / f;
        x *= inv; y *= inv;
        return *this;
    }
    
    __bidevice__ vec2<T> operator-(){
        return vec2<T>(-x, -y);
    }
    
    __bidevice__ vec2<T> operator+(const vec2<T> &v) const{
        return vec2<T>(x + v.x, y + v.y);
    }
    
    __bidevice__ vec2<T> operator+=(const vec2<T> &v){
        x += v.x; y += v.y;
        return *this;
    }
    
    __bidevice__ vec2<T> operator*(T s) const{
        return vec2<T>(x * s, y * s);
    }
    
    __bidevice__ vec2<T> &operator*=(T s){
        x *= s; y *= s;
        return *this;
    }
    
    __bidevice__ vec2<T> operator*(const vec2<T> &v) const{
        return vec2<T>(x * v.x, y * v.y);
    }
    
    __bidevice__ vec2<T> &operator*=(const vec2<T> &v){
        x *= v.x; y *= v.y;
        return *this;
    }
    
    __bidevice__ Float LengthSquared() const{ return x * x + y * y; }
    __bidevice__ Float Length() const{ return sqrt(LengthSquared()); }
};

template<typename T> class vec3{
    public:
    T x, y, z;
    
    __bidevice__ vec3(){ x = y = z = (T)0; }
    __bidevice__ vec3(T a){ x = y = z = a; }
    __bidevice__ vec3(T a, T b, T c): x(a), y(b), z(c){
        Assert(!HasNaN());
    }
    
    __bidevice__ bool HasNaN(){
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ bool IsBlack(){
        return (IsZero(x) && IsZero(y) && IsZero(z));
    }
    
    __bidevice__ bool IsBlack() const{
        return (IsZero(x) && IsZero(y) && IsZero(z));
    }
    
    __bidevice__ T operator[](int i) const{
        Assert(i >= 0 && i < 3);
        if(i == 0) return x;
        if(i == 1) return y;
        return z;
    }
    
    __bidevice__ T &operator[](int i){
        Assert(i >= 0 && i < 3);
        if(i == 0) return x;
        if(i == 1) return y;
        return z;
    }
    
    __bidevice__ vec3<T> operator/(T f) const{
        Assert(f != 0);
        Float inv = (Float)1 / f;
        return vec3<T>(x * inv, y * inv, z * inv);
    }
    
    __bidevice__ vec3<T> &operator/(T f){
        Assert(f != 0);
        Float inv = (Float)1 / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    
    __bidevice__ vec3<T> operator/(const vec3<T> &v) const{
        Assert(!v.HasNaN());
        Float invx = (Float)1 / v.x;
        Float invy = (Float)1 / v.y;
        Float invz = (Float)1 / v.z;
        return vec3<T>(x * invx, y * invy, z * invz);
    }
    
    __bidevice__ vec3<T> &operator/(const vec3<T> &v){
        Assert(!v.HasNaN());
        Float invx = (Float)1 / v.x;
        Float invy = (Float)1 / v.y;
        Float invz = (Float)1 / v.z;
        x = x * invx; y = y * invy; z = z * invz;
        return *this;
    }
    
    __bidevice__ vec3<T> operator-(){
        return vec3<T>(-x, -y, -z);
    }
    
    __bidevice__ vec3<T> operator-() const{
        return vec3<T>(-x, -y, -z);
    }
    
    __bidevice__ vec3<T> operator-(const vec3<T> &v) const{
        return vec3(x - v.x, y - v.y, z - v.z);
    }
    
    __bidevice__ vec3<T> &operator-=(const vec3<T> &v){
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    
    __bidevice__ vec3<T> operator+(const vec3<T> &v) const{
        return vec3<T>(x + v.x, y + v.y, z + v.z);
    }
    
    __bidevice__ vec3<T> operator+=(const vec3<T> &v){
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    
    __bidevice__ vec3<T> operator*(const vec3<T> &v) const{
        return vec3<T>(x * v.x, y * v.y, z * v.z);
    }
    
    __bidevice__ vec3<T> &operator*=(const vec3<T> &v){
        x *= v.x; y *= v.y; z *= v.z;
        return *this;
    }
    
    __bidevice__ vec3<T> operator*(T s) const{
        return vec3<T>(x * s, y * s, z * s);
    }
    
    __bidevice__ vec3<T> &operator*=(T s){
        x *= s; y *= s; z *= s;
        return *this;
    }
    
    __bidevice__ Float LengthSquared() const{ return x * x + y * y + z * z; }
    __bidevice__ Float Length() const{ return sqrt(LengthSquared()); }
};

template<typename T> inline __bidevice__ vec2<T> operator*(T s, vec2<T> &v){ return v * s; }
template<typename T> inline __bidevice__ vec3<T> operator*(T s, vec3<T> &v){ return v * s; }

template<typename T> inline __bidevice__ vec2<T> Abs(const vec2<T> &v){
    return vec2<T>(Absf(v.x), Absf(v.y));
}

template <typename T, typename U> inline __bidevice__ 
vec3<T> operator*(U s, const vec3<T> &v){
    return v * s;
}

template<typename T> inline vec3<T> __bidevice__ Abs(const vec3<T> &v){
    return vec3<T>(Absf(v.x), Absf(v.y), Absf(v.z));
}

template<typename T> inline T __bidevice__ Dot(const vec3<T> &v1, const vec3<T> &v2){
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template<typename T> inline __bidevice__ T AbsDot(const vec3<T> &v1, const vec3<T> &v2){
    return Absf(Dot(v1, v2));
}

template<typename T> inline __bidevice__ vec3<T> Cross(const vec3<T> &v1, const vec3<T> &v2){
    double v1x = v1.x, v1y = v1.y, v1z = v1.z;
    double v2x = v2.x, v2y = v2.y, v2z = v2.z;
    return vec3<T>((v1y * v2z) - (v1z * v2y),
                   (v1z * v2x) - (v1x * v2z),
                   (v1x * v2y) - (v1y * v2x));
}

template<typename T> inline __bidevice__ vec3<T> Normalize(const vec3<T> &v){
    return v / v.Length();
}

template<typename T> inline __bidevice__ T MinComponent(const vec3<T> &v){
    return Min(v.x, Min(v.y, v.z));
}

template<typename T> inline __bidevice__ T MaxComponent(const vec3<T> &v){
    return Max(v.x, Max(v.y, v.z));
}

template<typename T> inline __bidevice__ int MaxDimension(const vec3<T> &v){
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

template<typename T> inline __bidevice__ vec3<T> Min(const vec3<T> &v1, const vec3<T> &v2){
    return vec3<T>(Min(v1.x, v1.y), Min(v1.y, v2.y), Min(v1.z, v2.z));
}

template<typename T> inline __bidevice__ vec3<T> Max(const vec3<T> &v1, const vec3<T> &v2){
    return vec3<T>(Max(v1.x, v1.y), Max(v1.y, v2.y), Max(v1.z, v2.z));
}

template<typename T> inline __bidevice__ vec3<T> Permute(const vec3<T> &v, int x, int y, int z){
    return vec3<T>(v[x], v[y], v[z]);
}

template<typename T> inline __bidevice__ void 
CoordinateSystem(const vec3<T> &v1, vec3<T> *v2, vec3<T> *v3){
    if(Absf(v1.x) > Absf(v1.y)){
        *v2 = vec3<T>(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    }else{
        *v2 = vec3<T>(0, v1.z, -v1.y) / sqrt(v1.z * v1.z + v1.y * v1.y);
    }
    
    *v3 = Cross(v1, *v2);
}

template<typename T> inline __bidevice__
vec3<T> Sqrt(const vec3<T> &v){
    return vec3<T>(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z));
}

template<typename T> inline __bidevice__ 
vec3<T> Pow(const vec3<T> &v, Float val){
    return vec3<T>(std::pow(v.x, val), std::pow(v.y, val), std::pow(v.z, val));
}

template<typename T> inline __bidevice__
vec3<T> Exp(const vec3<T> &v){
    return vec3<T>(std::exp(v.x), std::exp(v.y), std::exp(v.z));
}

typedef vec2<Float> vec2f;
typedef vec2<int> vec2i;
typedef vec3<Float> vec3f;
typedef vec3<int> vec3i;

typedef vec3<Float> Spectrum; //RGB Spectrum

template<typename T> class Point3{
    public:
    T x, y, z;
    __bidevice__ Point3(){ x = y = z = (T)0; }
    __bidevice__ Point3(T a){ x = y = z = a; }
    __bidevice__ Point3(T a, T b, T c): x(a), y(b), z(c){
        Assert(!HasNaN());
    }
    
    template<typename U> explicit __bidevice__ Point3(const Point3<U> &p)
        :x((T)p.x), y((T)p.y), z((T)p.z)
    {
        Assert(!HasNaN());
    }
    
    template<typename U> explicit __bidevice__ operator vec3<U>() const{
        return vec3<U>(x, y, z);
    }
    
    __bidevice__ Point3<T> operator/(T f) const{
        Assert(f != 0);
        Float inv = (Float)1 / f;
        return Point3<T>(x * inv, y * inv, z * inv);
    }
    
    __bidevice__ Point3<T> &operator/(T f){
        Assert(f != 0);
        Float inv = (Float)1 / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    
    __bidevice__ Point3<T> operator*(T s) const{
        return Point3<T>(x * s, y * s, z * s);
    }
    
    __bidevice__ Point3<T> &operator*=(T s){
        x *= s; y *= s; z *= s;
        return *this;
    }
    
    __bidevice__ Point3<T> operator*(const Point3<T> &s) const{
        return Point3<T>(x * s.x, y * s.y, z * s.z);
    }
    
    __bidevice__ Point3<T> &operator*=(const Point3<T> &s){
        x *= s.x; y *= s.y; z *= s.z;
        return *this;
    }
    
    __bidevice__ Point3<T> operator+(const Point3<T> &p) const{
        return Point3<T>(x + p.x, y + p.y, z + p.z);
    }
    
    __bidevice__ Point3<T> operator+=(const Point3<T> &p){
        x += p.x; y += p.y; z += p.z;
        return *this;
    }
    
    __bidevice__ Point3<T> operator+(const vec3<T> &v) const{
        return Point3<T>(x + v.x, y + v.y, z + v.z);
    }
    
    __bidevice__ Point3<T> &operator+=(const vec3<T> &v){
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    
    __bidevice__ vec3<T> operator-(const Point3<T> &p) const{
        return vec3<T>(x - p.x, y - p.y, z - p.z);
    }
    
    __bidevice__ Point3<T> operator-(const vec3<T> &v) const{
        return Point3<T>(x - v.x, y - v.y, z - v.z);
    }
    
    __bidevice__ Point3<T> &operator-=(const vec3<T> &v){
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    
    __bidevice__ T operator[](int i) const {
        Assert(i >= 0 && i < 3);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    __bidevice__ T &operator[](int i) {
        Assert(i >= 0 && i < 3);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
};

template <typename T, typename U> inline __bidevice__ 
Point3<T> operator*(U s, const Point3<T> &v){
    return v * s;
}

template<typename T> class Point2{
    public:
    T x, y;
    __bidevice__ Point2(){ x = y = (T)0; }
    __bidevice__ Point2(T a, T b): x(a), y(b){
        Assert(!HasNaN());
    }
    
    __bidevice__ explicit Point2(const Point3<T> &p): x(p.x), y(p.y){
        Assert(!HasNaN());
    }
    
    __bidevice__ Point2<T> operator*(T s) const{
        return Point2<T>(x * s, y * s);
    }
    
    __bidevice__ Point2<T> &operator*=(T s){
        x *= s; y *= s;
        return *this;
    }
    
    __bidevice__ Point2<T> operator*(const vec2<T> &v) const{
        return Point2<T>(x * v.x, y * v.y);
    }
    
    __bidevice__ Point2<T> &operator*=(const vec2<T> &v){
        x *= v.x; y *= v.y;
        return *this;
    }
    
    __bidevice__ Point2<T> operator*(const Point2<T> &v) const{
        return Point2<T>(x * v.x, y * v.y);
    }
    
    __bidevice__ Point2<T> &operator*=(const Point2<T> &v){
        x *= v.x; y *= v.y;
        return *this;
    }
    
    __bidevice__ Point2<T> operator+(const Point2<T> &p) const{
        return Point2<T>(x + p.x, y + p.y);
    }
    
    __bidevice__ Point2<T> operator+=(const Point2<T> &p){
        x += p.x; y += p.y;
        return *this;
    }
    
    __bidevice__ Point2<T> operator+(const vec2<T> &v) const{
        return Point2<T>(x + v.x, y + v.y);
    }
    
    __bidevice__ Point2<T> &operator+=(const vec2<T> &v){
        x += v.x; y += v.y;
        return *this;
    }
    
    __bidevice__ vec2<T> operator-(const Point2<T> &p) const{
        return vec2<T>(x - p.x, y - p.y);
    }
    
    __bidevice__ Point2<T> operator-(const vec2<T> &v) const{
        return Point2<T>(x - v.x, y - v.y);
    }
    
    __bidevice__ Point2<T> &operator-=(const vec2<T> &v){
        x -= v.x; y -= v.y;
        return *this;
    }
    
    __bidevice__ T operator[](int i) const {
        Assert(i >= 0 && i < 2);
        if (i == 0) return x;
        return y;
    }
    
    __bidevice__ T &operator[](int i) {
        Assert(i >= 0 && i < 2);
        if (i == 0) return x;
        return y;
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y);
    }
};

template<typename T> inline __bidevice__ Float Distance(const Point3<T> &p1, 
                                                        const Point3<T> &p2)
{
    return (p1 - p2).Length();
}

template<typename T> inline __bidevice__ Float Distance(const Point2<T> &p1, 
                                                        const Point2<T> &p2)
{
    return (p1 - p2).Length();
}

template<typename T> inline __bidevice__ Float DistanceSquared(const Point3<T> &p1, 
                                                               const Point3<T> &p2)
{
    return (p1 - p2).LengthSquared();
}

template<typename T> inline __bidevice__ Float DistanceSquared(const Point2<T> &p1, 
                                                               const Point2<T> &p2)
{
    return (p1 - p2).LengthSquared();
}

template<typename T> inline __bidevice__ Point3<T> Lerp(Float t, const Point3<T> &p0, 
                                                        const Point3<T> &p1)
{
    return (1 - t) * p0 + t * p1;
}

template<typename T> inline __bidevice__ Point3<T> Min(const Point3<T> &p0, 
                                                       const Point3<T> &p1)
{
    return Point3<T>(Min(p0.x, p1.x), Min(p0.y, p1.y), Min(p0.z, p1.z));
}

template<typename T> inline __bidevice__ Point3<T> Max(const Point3<T> &p0, 
                                                       const Point3<T> &p1)
{
    return Point3<T>(Max(p0.x, p1.x), Max(p0.y, p1.y), Max(p0.z, p1.z));
}

inline __bidevice__ Float Floor(const Float &v){
    return std::floor(v);
}

template<typename T> inline __bidevice__ Point3<T> Floor(const Point3<T> &p0)
{
    return Point3<T>(floor(p0.x), floor(p0.y), floor(p0.z));
}

template<typename T> inline __bidevice__ Point3<T> Ceil(const Point3<T> &p0){
    return Point3<T>(ceil(p0.x), ceil(p0.y), ceil(p0.z));
}

template<typename T> inline __bidevice__ Point3<T> Abs(const Point3<T> &p0){
    return Point3<T>(Absf(p0.x), Absf(p0.y), Absf(p0.z));
}

template<typename T> inline __bidevice__ Point3<T> Permute(const Point3<T> &p0, int x, int y, int z){
    return Point3<T>(p0[x], p0[y], p0[z]);
}

template <typename T, typename U> inline __bidevice__ 
Point2<T> operator*(U s, const Point2<T> &v){
    return v * s;
}

typedef Point2<Float> Point2f;
typedef Point2<int> Point2i;
typedef Point3<Float> Point3f;
typedef Point3<int> Point3i;

template<typename T> class Normal3{
    public:
    T x, y, z;
    __bidevice__ Normal3(){ x = y = z = (T)0; }
    __bidevice__ Normal3(T a){ x = y = z = a; }
    __bidevice__ Normal3(T a, T b, T c): x(a), y(b), z(c)
    {
        Assert(!HasNaN());
    }
    
    __bidevice__ bool HasNaN(){
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ bool HasNaN() const{
        return IsNaN(x) || IsNaN(y) || IsNaN(z);
    }
    
    __bidevice__ Normal3<T> operator-() const { return Normal3(-x, -y, -z); }
    __bidevice__ Normal3<T> operator+(const Normal3<T> &n) const {
        Assert(!n.HasNaN());
        return Normal3<T>(x + n.x, y + n.y, z + n.z);
    }
    
    __bidevice__ Normal3<T> &operator+=(const Normal3<T> &n) {
        Assert(!n.HasNaN());
        x += n.x; y += n.y; z += n.z;
        return *this;
    }
    __bidevice__ Normal3<T> operator-(const Normal3<T> &n) const {
        Assert(!n.HasNaN());
        return Normal3<T>(x - n.x, y - n.y, z - n.z);
    }
    
    __bidevice__ Normal3<T> &operator-=(const Normal3<T> &n) {
        Assert(!n.HasNaN());
        x -= n.x; y -= n.y; z -= n.z;
        return *this;
    }
    
    template <typename U> __bidevice__ Normal3<T> operator*(U f) const {
        return Normal3<T>(f * x, f * y, f * z);
    }
    
    template <typename U> __bidevice__ Normal3<T> &operator*=(U f) {
        x *= f; y *= f; z *= f;
        return *this;
    }
    template <typename U> __bidevice__ Normal3<T> operator/(U f) const {
        Assert(f != 0);
        Float inv = (Float)1 / f;
        return Normal3<T>(x * inv, y * inv, z * inv);
    }
    
    template <typename U> __bidevice__ Normal3<T> &operator/=(U f) {
        Assert(f != 0);
        Float inv = (Float)1 / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    
    __bidevice__ explicit Normal3<T>(const vec3<T> &v) : x(v.x), y(v.y), z(v.z) {}
    
    __bidevice__ bool operator==(const Normal3<T> &n) const {
        return x == n.x && y == n.y && z == n.z;
    }
    
    __bidevice__ bool operator!=(const Normal3<T> &n) const {
        return x != n.x || y != n.y || z != n.z;
    }
    
    __bidevice__ T operator[](int i) const {
        Assert(i >= 0 && i < 3);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    __bidevice__ T &operator[](int i) {
        Assert(i >= 0 && i < 3);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    
    __bidevice__ Float LengthSquared() const { return x * x + y * y + z * z; }
    __bidevice__ Float Length() const { return sqrt(LengthSquared()); }
};

template <typename T> inline __bidevice__ Normal3<T> Normalize(const Normal3<T> &n) {
    return n / n.Length();
}

template <typename T>
inline __bidevice__ Normal3<T> Faceforward(const Normal3<T> &n, const vec3<T> &v) {
    return (Dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
inline __bidevice__ Normal3<T> Faceforward(const Normal3<T> &n, const Normal3<T> &n2) {
    return (Dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
inline __bidevice__ vec3<T> Faceforward(const vec3<T> &v, const vec3<T> &v2) {
    return (Dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
inline __bidevice__ vec3<T> Faceforward(const vec3<T> &v, const Normal3<T> &n2) {
    return (Dot(v, n2) < 0.f) ? -v : v;
}

template <typename T>
inline __bidevice__ Normal3<T> Abs(const Normal3<T> &v) {
    return Normal3<T>(Absf(v.x), Absf(v.y), Absf(v.z));
}

template <typename T> inline __bidevice__ T Dot(const Normal3<T> &n1, const Normal3<T> &n2) {
    return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}

template <typename T> inline __bidevice__ T Dot(const Normal3<T> &n1, const vec3<T> &v) {
    return n1.x * v.x + n1.y * v.y + n1.z * v.z;
}

template <typename T> inline __bidevice__ T AbsDot(const Normal3<T> &n1, const vec3<T> &v2) {
    return Absf(n1.x * v2.x + n1.y * v2.y + n1.z * v2.z);
}

template<typename T> inline __bidevice__ vec3<T> ToVec3(const Normal3<T> &n){
    return vec3<T>(n.x, n.y, n.z);
}

template<typename T> inline __bidevice__ vec3<T> ToVec3(const Point3<T> &p){
    return vec3<T>(p.x, p.y, p.z);
}

template<typename T> inline __bidevice__ Normal3<T> toNormal3(const vec3<T> &v){
    return Normal3<T>(v.x, v.y, v.z);
}

typedef Normal3<Float> Normal3f;

class Ray{
    public:
    Point3f o;
    vec3f d;
    mutable Float tMax;
    Float time;
    
    __bidevice__ Ray() : tMax(Infinity), time(0.f) {}
    __bidevice__ Ray(const Point3f &o, const vec3f &d, Float tMax = Infinity,
                     Float time = 0.f)
        : o(o), d(d), tMax(tMax), time(time) {}
    
    __bidevice__ Point3f operator()(Float t){ return o + d * t; }
    __bidevice__ Point3f operator()(Float t) const{ return o + d * t; }
};

class RayDifferential : public Ray{
    public:
    bool hasDifferentials;
    Point3f rxOrigin, ryOrigin;
    vec3f rxDirection, ryDirection;
    
    __bidevice__ RayDifferential() { hasDifferentials = false; }
    __bidevice__ RayDifferential(const Point3f &o, const vec3f &d, 
                                 Float tMax = Infinity, Float time = 0.f)
        : Ray(o, d, tMax, time) 
    {
        hasDifferentials = false;
    }
    
    __bidevice__ RayDifferential(const Ray &ray) : Ray(ray) { hasDifferentials = false; }
    
    __bidevice__ void ScaleDifferentials(Float s) {
        rxOrigin = o + (rxOrigin - o) * s;
        ryOrigin = o + (ryOrigin - o) * s;
        rxDirection = d + (rxDirection - d) * s;
        ryDirection = d + (ryDirection - d) * s;
    }
};


template <typename T>
class Bounds3 {
    public:
    Point3<T> pMin, pMax;
    
    __bidevice__ Bounds3(){
        T minNum = FLT_MIN;
        T maxNum = FLT_MAX;
        pMin = Point3<T>(maxNum, maxNum, maxNum);
        pMax = Point3<T>(minNum, minNum, minNum);
    }
    
    __bidevice__ explicit Bounds3(const Point3<T> &p) : pMin(p), pMax(p) {}
    __bidevice__ Bounds3(const Point3<T> &p1, const Point3<T> &p2)
        : pMin(Min(p1.x, p2.x), Min(p1.y, p2.y),
               Min(p1.z, p2.z)),
    pMax(Max(p1.x, p2.x), Max(p1.y, p2.y),
         Max(p1.z, p2.z)) {}
    
    __bidevice__ const Point3<T> &operator[](int i) const;
    __bidevice__ Point3<T> &operator[](int i);
    __bidevice__ bool operator==(const Bounds3<T> &b) const{
        return b.pMin == pMin && b.pMax == pMax;
    }
    
    __bidevice__ bool operator!=(const Bounds3<T> &b) const{
        return b.pMin != pMin || b.pMax != pMax;
    }
    
    __bidevice__ Point3<T> Corner(int corner) const{
        Assert(corner >= 0 && corner < 8);
        return Point3<T>((*this)[(corner & 1)].x,
                         (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }
    
    __bidevice__ vec3<T> Diagonal() const { return pMax - pMin; }
    __bidevice__ T SurfaceArea() const{
        vec3<T> d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    
    __bidevice__ T Volume() const{
        vec3<T> d = Diagonal();
        return d.x * d.y * d.z;
    }
    
    __bidevice__ T ExtentOn(int i) const{
        Assert(i >= 0 && i < 3);
        if(i == 0) return Absf(pMax.x - pMin.x);
        if(i == 1) return Absf(pMax.y - pMin.y);
        return Absf(pMax.z - pMin.z);
    }
    
    __bidevice__ int MaximumExtent() const{
        vec3<T> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
    
    __bidevice__ Point3<T> Lerp(const Point3f &t) const{
        return Point3<T>(Lerp(t.x, pMin.x, pMax.x),
                         Lerp(t.y, pMin.y, pMax.y),
                         Lerp(t.z, pMin.z, pMax.z));
    }
    
    __bidevice__ vec3<T> Offset(const Point3<T> &p) const{
        vec3<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }
    
    __bidevice__ void BoundingSphere(Point3<T> *center, Float *radius) const{
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }
    
    template <typename U> __bidevice__ explicit operator Bounds3<U>() const{
        return Bounds3<U>((Point3<U>)pMin, (Point3<U>)pMax);
    }
    
    __bidevice__ bool IntersectP(const Ray &ray, Float *hitt0 = nullptr,
                                 Float *hitt1 = nullptr) const;
    
    inline __bidevice__ bool IntersectP(const Ray &ray, const vec3f &invDir,
                                        const int dirIsNeg[3]) const;
    
    __bidevice__ void PrintSelf() const{
        printf("pMin = {x : %g, y : %g, z : %g} pMax = {x : %g, y : %g, z : %g}",
               pMin.x, pMin.y, pMin.z, pMax.x, pMax.y, pMax.z);
    }
};

typedef Bounds3<Float> Bounds3f;
typedef Bounds3<int> Bounds3i;


template <typename T> inline __bidevice__ 
Point3<T> &Bounds3<T>::operator[](int i){
    Assert(i == 0 || i == 1);
    return (i == 0) ? pMin : pMax;
}

template <typename T> inline __bidevice__
Bounds3<T> Union(const Bounds3<T> &b, const Point3<T> &p){
    Bounds3<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T> inline __bidevice__
Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2){
    Bounds3<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T> inline __bidevice__ 
Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2){
    Bounds3<T> ret;
    ret.pMin = Max(b1.pMin, b2.pMin);
    ret.pMax = Min(b1.pMax, b2.pMax);
    return ret;
}

template <typename T> inline __bidevice__ 
bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2){
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

template <typename T> inline __bidevice__
bool Inside(const Point3<T> &p, const Bounds3<T> &b){
    return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
            p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
}

template <typename T> inline __bidevice__
bool InsideExclusive(const Point3<T> &p, const Bounds3<T> &b){
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y &&
            p.y < b.pMax.y && p.z >= b.pMin.z && p.z < b.pMax.z);
}

template <typename T, typename U> inline __bidevice__ 
Bounds3<T> Expand(const Bounds3<T> &b, U delta){
    return Bounds3<T>(b.pMin - vec3<T>(delta, delta, delta),
                      b.pMax + vec3<T>(delta, delta, delta));
}

inline __bidevice__ 
vec3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi){
    return vec3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

inline __bidevice__
vec3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi, 
                         const vec3f &x, const vec3f &y, const vec3f &z)
{
    return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y + cosTheta * z;
}

template <typename T> inline __bidevice__ 
bool Bounds3<T>::IntersectP(const Ray &ray, Float *hitt0, Float *hitt1) const{
    Float t0 = 0, t1 = ray.tMax;
    for (int i = 0; i < 3; ++i) {
        Float invRayDir = 1 / ray.d[i];
        Float tNear = (pMin[i] - ray.o[i]) * invRayDir;
        Float tFar = (pMax[i] - ray.o[i]) * invRayDir;
        
        if (tNear > tFar) swap(tNear, tFar);
        
        tFar *= 1 + 2 * gamma(3);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) return false;
    }
    if (hitt0) *hitt0 = t0;
    if (hitt1) *hitt1 = t1;
    return true;
}

template <typename T> inline __bidevice__ 
bool Bounds3<T>::IntersectP(const Ray &ray, const vec3f &invDir,
                            const int dirIsNeg[3]) const 
{
    const Bounds3f &bounds = *this;
    Float tMin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
    Float tMax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
    Float tyMin = (bounds[dirIsNeg[1]].y - ray.o.y) * invDir.y;
    Float tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;
    
    tMax *= 1 + 2 * gamma(3);
    tyMax *= 1 + 2 * gamma(3);
    if (tMin > tyMax || tyMin > tMax) return false;
    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMax) tMax = tyMax;
    
    Float tzMin = (bounds[dirIsNeg[2]].z - ray.o.z) * invDir.z;
    Float tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;
    
    tzMax *= 1 + 2 * gamma(3);
    if (tMin > tzMax || tzMin > tMax) return false;
    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMax) tMax = tzMax;
    return (tMin < ray.tMax) && (tMax > 0);
}

//NOTE: Must be normalized
inline __bidevice__
Float SphericalTheta(const vec3f &v){
    return std::acos(Clamp(v.z, -1, 1));
}

inline __bidevice__
Float SphericalPhi(const vec3f &v){
    Float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Pi) : p;
}

inline __bidevice__ 
bool SameHemisphere(const vec3f &w, const vec3f &wp) {
    return w.z * wp.z > 0;
}

inline __bidevice__ 
bool SameHemisphere(const vec3f &w, const Normal3f &wp) {
    return w.z * wp.z > 0;
}

inline __bidevice__ 
Point2f ConcentricSampleDisk(const Point2f &u){
    Point2f uOffset = 2.f * u - vec2f(1, 1);
    
    if(uOffset.x == 0 && uOffset.y == 0) return Point2f(0, 0);
    
    Float theta, r;
    if(Absf(uOffset.x) > Absf(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * Point2f(std::cos(theta), std::sin(theta));
}

inline __bidevice__ 
vec3f CosineSampleHemisphere(const Point2f &u) {
    Point2f d = ConcentricSampleDisk(u);
    Float z = std::sqrt(Max((Float)0, 1 - d.x * d.x - d.y * d.y));
    return vec3f(d.x, d.y, z);
}

//Centered at +Z
inline __bidevice__ Float CosTheta(const vec3f &w) { return w.z; }
inline __bidevice__ Float Cos2Theta(const vec3f &w) { return w.z * w.z; }
inline __bidevice__ Float AbsCosTheta(const vec3f &w) { return Absf(w.z); }
inline __bidevice__ Float Sin2Theta(const vec3f &w){
    return Max((Float)0, (Float)1 - Cos2Theta(w));
}

inline __bidevice__ Float SinTheta(const vec3f &w) { return std::sqrt(Sin2Theta(w)); }

inline __bidevice__ Float TanTheta(const vec3f &w) { return SinTheta(w) / CosTheta(w); }

inline __bidevice__ Float Tan2Theta(const vec3f &w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

inline __bidevice__ Float CosPhi(const vec3f &w){
    Float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
}

inline __bidevice__ Float SinPhi(const vec3f &w){
    Float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
}

inline __bidevice__ Float Cos2Phi(const vec3f &w) { return CosPhi(w) * CosPhi(w); }

inline __bidevice__ Float Sin2Phi(const vec3f &w) { return SinPhi(w) * SinPhi(w); }

inline __bidevice__ vec3f Reflect(const vec3f &wo, const vec3f &n){
    return -wo + 2 * Dot(wo, n) * n;
}

inline __bidevice__ bool Refract(const vec3f &wi, const Normal3f &n, Float eta, vec3f *wt){
    Float cosThetaI  = Dot(n, wi);
    Float sin2ThetaI = Max(Float(0), Float(1 - cosThetaI * cosThetaI));
    Float sin2ThetaT = eta * eta * sin2ThetaI;
    
    if(sin2ThetaT >= 1) return false;
    Float cosThetaT = std::sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * ToVec3(n);
    return true;
}