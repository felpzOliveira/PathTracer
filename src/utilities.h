#if !defined(UTILITIES_H)
#define UTILITIES_H
#include <consts.h>
#include <time.h>
#include <cutil.h>
#include <glm/glm.hpp>

#define ABS(x) ((x) > 0.0001 ? (x) : -(x))

inline __bidevice__ float gamma(int n){
    float MachineEpsilon = 5.96046e-08f;
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

inline __bidevice__ bool IsNaN(float v){
    if(v != v) return true;
    return false;
}

template<typename T>
inline __bidevice__ bool IsNaN(T v){
    return (IsNaN(v.x) || IsNaN(v.y) || IsNaN(v.z));
}

inline __bidevice__ bool IsZero(float v){
    return (ABS(v) < 0.0001f);
}

inline __bidevice__ float Clamp(float x, float a, float b){
    if(x < a) return a;
    if(x > b) return b;
    return x;
}

inline __bidevice__ float clamp(float x, float a, float b){
    return Clamp(x, a, b);
}

inline __host__ float random_float(){
    return (rand() / (RAND_MAX + 1.0f));
}

inline __bidevice__ glm::vec3 abs(glm::vec3 v){
    return glm::vec3(ABS(v.x), ABS(v.y), ABS(v.z));
}

inline __bidevice__ glm::vec3 Faceforward(const glm::vec3 &n, 
                                          const glm::vec3 &v) 
{
    return (glm::dot(n, v) < 0.f) ? -n : n;
}

inline __bidevice__ float AbsDot(glm::vec3 v0, glm::vec3 v1){
    return ABS(glm::dot(v0, v1));
}

inline __bidevice__ float Rclamp(float a, float b, float x){
    if(x < a) return a;
    if(x > b) return b;
    return x;
}

inline __bidevice__ float CosTheta(glm::vec3 w){ return w.z; }

inline __bidevice__ float Cos2Theta(glm::vec3 w){ return w.z * w.z; }

inline __bidevice__ float AbsCosTheta(glm::vec3 w){ return glm::abs(w.z); }

inline __bidevice__ float Sin2Theta(glm::vec3 w){ return glm::max(0.0f, 1.0f - Cos2Theta(w)); 
}

inline __bidevice__ float SinTheta(glm::vec3 w){ return glm::sqrt(Sin2Theta(w)); }

inline __bidevice__ float TanTheta(glm::vec3 w){ return SinTheta(w) / CosTheta(w); }

inline __bidevice__ float Tan2Theta(glm::vec3 w){ return Sin2Theta(w) / Cos2Theta(w); }

inline __bidevice__ float CosPhi(glm::vec3 w){ 
    float s = SinTheta(w);
    return (glm::abs(s) < 0.0001f) ? 0.0f : Rclamp(-1.0f, 1.0f, w.x/s);
}

inline __bidevice__ float SinPhi(glm::vec3 w){ 
    float s = SinTheta(w);
    return (glm::abs(s) < 0.0001f) ? 0.0f : Rclamp(-1.0f, 1.0f, w.y/s);
}

inline __bidevice__ float Cos2Phi(glm::vec3 w){ return CosPhi(w) * CosPhi(w); }

inline __bidevice__ float Sin2Phi(glm::vec3 w){ return SinPhi(w) * SinPhi(w); }

inline __bidevice__ float CosDPhi(glm::vec3 wa, glm::vec3 wb) {
    return Rclamp(-1.0f, 1.0f, 
                  (wa.x * wb.x + wa.y * wb.y) /
                  glm::sqrt((wa.x * wa.x + wa.y * wa.y) *
                            (wb.x * wb.x + wb.y * wb.y)));
}

inline __bidevice__ float ErfInv(float x) {
    float w, p;
    x = Clamp(x, -.99999f, .99999f);
    w = -glm::log((1 - x) * (1 + x));
    if (w < 5) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = glm::sqrt(w) - 3;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

inline __bidevice__  float Erf(float x) {
    // constants
    float a1 = 0.254829592f;
    float a2 = -0.284496736f;
    float a3 = 1.421413741f;
    float a4 = -1.453152027f;
    float a5 = 1.061405429f;
    float p = 0.3275911f;
    
    // Save the sign of x
    int sign = 1;
    if (x < 0) sign = -1;
    x = ABS(x);
    
    // A&S formula 7.1.26
    float t = 1 / (1 + p * x);
    float y =
        1 -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * glm::exp(-x * x);
    
    return sign * y;
}

inline __bidevice__
glm::vec2 ConcentricSampleDisk(const glm::vec2 &u) {
    // Map uniform random numbers to [-1,1]
    glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1); 
    
    // Handle degeneracy at the origin
    if (ABS(uOffset.x) < 0.0001 && ABS(uOffset.y) < 0.0001) return glm::vec2(0, 0); 
    
    // Apply concentric mapping to point
    float theta, r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }   
    return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}

inline __bidevice__
glm::vec3 CosineSampleHemisphere(const glm::vec2 &u) {
    glm::vec2 d = ConcentricSampleDisk(u);
    float z = glm::sqrt(glm::max((float)0, 1 - d.x * d.x - d.y * d.y));
    return glm::vec3(d.x, d.y, z); 
}

inline __bidevice__ bool SameHemisphere(const glm::vec3 &w, const glm::vec3 &wp) {
    return w.z * wp.z > 0;
}


inline __bidevice__
glm::vec3 SphericalDirection(float sinTheta, float cosTheta, float phi){
    return glm::vec3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi),
                     cosTheta);
}

inline __bidevice__
glm::vec3 SphericalDirection(float sinTheta, float cosTheta, float phi,
                             const glm::vec3 &x, const glm::vec3 &y,
                             const glm::vec3 &z) 
{
    return sinTheta * glm::cos(phi) * x + sinTheta * glm::sin(phi) * y +
        cosTheta * z;
}


inline __bidevice__
float DistanceSquared(glm::vec3 v0, glm::vec3 v1){
    float dx = (v0.x - v1.x);
    float dy = (v0.y - v1.y);
    float dz = (v0.z - v1.z);
    return dx * dx + dy * dy + dz * dz;
}

inline __bidevice__
glm::vec3 UniformSampleHemisphere(const glm::vec2 &u) {
    float z = u[0];
    float r = glm::sqrt(glm::max((float)0, (float)1. - z * z));
    float phi = 2 * Pi * u[1];
    return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), z); 
}

inline __bidevice__
float UniformHemispherePdf() { return Inv2Pi; }

inline __bidevice__
glm::vec3 UniformSampleSphere(const glm::vec2 &u) {
    float z = 1 - 2 * u[0];
    float r = glm::sqrt(glm::max((float)0, (float)1 - z * z));
    float phi = 2 * Pi * u[1];
    return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), z); 
}

inline __bidevice__
float UniformSpherePdf() { return Inv4Pi; }

inline __bidevice__
float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf, g = ng * gPdf;
    return (f * f) / (f * f + g * g); 
}

inline __bidevice__
float LengthSquared(glm::vec3 v){
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

#endif