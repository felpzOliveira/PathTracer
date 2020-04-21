#if !defined(SPECTRUM_H)
#define SPECTRUM_H
#include <glm/glm.hpp>
#include <math.h>
#include <stdio.h>
#include <cuda_util.cuh>

enum class SpectrumType { Reflectance, Illuminant };

__host__ __device__ 
inline void XYZToRGB(const float xyz[3], float rgb[3]) {
    rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
    rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
    rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
}

__host__ __device__ 
inline void RGBToXYZ(const float rgb[3], float xyz[3]) {
    xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
    xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
    xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
}

template<int nSpectrumSamples> class CoefficientSpectrum : public Managed{
    public:
    static const int nSamples = nSpectrumSamples;
    
    protected:
    float c[nSpectrumSamples];
    
    public:
    
    __host__ __device__ CoefficientSpectrum(float v = 0.0f){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] = v;
        }
    }
    
    __host__ __device__ CoefficientSpectrum &operator=(const CoefficientSpectrum &s) 
    {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] = s.c[i];
        return *this;
    }
    
    __host__ __device__ 
        CoefficientSpectrum &operator+=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] += s2.c[i];
        }
        
        return *this;
    }
    
    __host__ __device__ 
        CoefficientSpectrum operator+(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] += s2.c[i];
        }
        
        return ret;
    }
    
    __host__ __device__ 
        CoefficientSpectrum &operator-=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] -= s2.c[i];
        }
        
        return *this;
    }
    
    __host__ __device__ 
        CoefficientSpectrum operator-(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] -= s2.c[i];
        }
        
        return ret;
    }
    
    __host__ __device__ 
        CoefficientSpectrum &operator*=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] *= s2.c[i];
        }
        
        return *this;
    }
    
    __host__ __device__ 
        CoefficientSpectrum operator*(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] *= s2.c[i];
        }
        
        return ret;
    }
    
    __host__ __device__ 
        CoefficientSpectrum &operator/=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] /= s2.c[i];
            if(c[i] != c[i]){
                printf("Division by 0 on spectrum [%d]?\n", i);
            }
        }
        
        return *this;
    }
    
    __host__ __device__ 
        CoefficientSpectrum operator/(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] /= s2.c[i];
            if(ret.c[i] != ret.c[i]){
                printf("Division by 0 on spectrum [%d]?\n", i);
            }
        }
        
        return ret;
    }
    
    __host__ __device__ 
        CoefficientSpectrum operator*(float a) const { 
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] *= a;
        return ret;
    }
    
    __host__ __device__ 
        CoefficientSpectrum &operator*=(float a) { 
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] *= a;
        return *this; 
    }
    
    __host__ __device__ 
        friend inline CoefficientSpectrum operator*(float a,const CoefficientSpectrum &s)
    {
        return s * a;
    }
    
    __host__ __device__ 
        CoefficientSpectrum operator/(float a) const { 
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i){
            ret.c[i] /= a;
            if(ret.c[i] != ret.c[i]){
                printf("Division by 0 on spectrum [%d]?\n", i);
            }
        }
        
        return ret;
    }
    
    __host__ __device__ 
        CoefficientSpectrum &operator/=(float a) { 
        for (int i = 0; i < nSpectrumSamples; ++i){
            c[i] /= a;
            
            if(c[i] != c[i]){
                printf("Division by 0 on spectrums?\n");
            }
        }
        
        return *this;
    }
    
    __host__ __device__ 
        bool operator==(const CoefficientSpectrum &sp) const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != sp.c[i]) return false;
        return true;
    }
    
    __host__ __device__ 
        bool operator!=(const CoefficientSpectrum &sp) const {
        return !(*this == sp);
    }
    
    __host__ __device__ 
        bool IsBlack() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != 0.) return false;
        return true;
    }
    
    __host__ __device__ 
        friend CoefficientSpectrum Sqrt(const CoefficientSpectrum &s) { 
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = glm::sqrt(s.c[i]);
        return ret;
    }
    
    template <int n> __host__ __device__  friend inline CoefficientSpectrum<n> Pow(const CoefficientSpectrum<n> &s, float e);
    
    __host__ __device__ 
        CoefficientSpectrum operator-() const {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = -c[i];
        return ret;
    }
    
    __host__ __device__ 
        friend CoefficientSpectrum Exp(const CoefficientSpectrum &s) {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = glm::exp(s.c[i]);
        return ret;
    }
    
    __host__ __device__ 
        CoefficientSpectrum Clamp(float low = 0, float high = FLT_MAX) const {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = clamp(low, high, c[i]);
        return ret;
    }
    
    __host__ __device__ 
        bool HasNaNs() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != c[i]) return true;
        return false;
    }
    
    __host__ __device__ 
        float &operator[](int i) {
        return c[i];
    }
    
    __host__ __device__ 
        float operator[](int i) const {
        return c[i];
    }
};


class RGBSpectrum : public CoefficientSpectrum<3>{
    public:
    __host__ __device__ RGBSpectrum(float v = 0.f) : CoefficientSpectrum<3>(v) { }
    
    __host__ __device__ RGBSpectrum(const CoefficientSpectrum<3> &v) :
    CoefficientSpectrum<3>(v) { }
    
    __host__ __device__ RGBSpectrum(const RGBSpectrum &s){
        *this = s;
    }
    
    __host__ __device__ static RGBSpectrum FromRGB(float r, float g, float b){
        RGBSpectrum s;
        s.c[0] = r;
        s.c[1] = g;
        s.c[2] = b;
        return s;
    }
    
    __host__ __device__ static RGBSpectrum FromRGB(const float rgb[3],
                                                   SpectrumType type =
                                                   SpectrumType::Reflectance) 
    {
        RGBSpectrum s;
        s.c[0] = rgb[0];
        s.c[1] = rgb[1];
        s.c[2] = rgb[2];
        return s;
    }
    
    __host__ __device__ 
        void ToRGB(glm::vec3 *v) const {
        *v = glm::vec3(c[0], c[1], c[2]);
    }
    
    __host__ __device__ 
        void ToRGB(float *rgb) const {
        rgb[0] = c[0];
        rgb[1] = c[1];
        rgb[2] = c[2];
    }
    
    __host__ __device__ 
        const RGBSpectrum &ToRGBSpectrum() const {
        return *this;
    }
    
    __host__ __device__ 
        void ToXYZ(float xyz[3]) const { 
        RGBToXYZ(c, xyz);
    }
    
    __host__ __device__ 
        static RGBSpectrum FromXYZ(const float xyz[3],
                                   SpectrumType type = SpectrumType::Reflectance) {
        RGBSpectrum r;
        XYZToRGB(xyz, r.c);
        return r;
    }
    
    __host__ __device__ 
        float y() const {
        const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
        return YWeight[0] * c[0] + YWeight[1] * c[1] + YWeight[2] * c[2];
    }
};

typedef RGBSpectrum Spectrum;

#endif