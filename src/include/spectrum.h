#if !defined(SPECTRUM_H)
#define SPECTRUM_H
#include <glm/glm.hpp>
#include <stdio.h>
#include <math.h>
#include <cutil.h>
#include <utilities.h>

enum class SpectrumType { Reflectance, Illuminant };

__bidevice__ 
inline void XYZToRGB(const float xyz[3], float rgb[3]) {
    rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
    rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
    rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
}

__bidevice__ 
inline void RGBToXYZ(const float rgb[3], float xyz[3]) {
    xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
    xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
    xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
}

template<int nSpectrumSamples> class CoefficientSpectrum : public Managed{
    public:
    static const int nSamples = nSpectrumSamples;
    
    float c[nSpectrumSamples];
    
    public:
    
    __bidevice__ CoefficientSpectrum(float v = 0.0f){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] = v;
        }
    }
    
    __bidevice__ CoefficientSpectrum &operator=(const CoefficientSpectrum &s) 
    {
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] = s.c[i];
        return *this;
    }
    
    __bidevice__ 
        CoefficientSpectrum &operator+=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] += s2.c[i];
        }
        
        return *this;
    }
    
    __bidevice__ 
        CoefficientSpectrum operator+(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] += s2.c[i];
        }
        
        return ret;
    }
    
    __bidevice__ 
        CoefficientSpectrum &operator-=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] -= s2.c[i];
        }
        
        return *this;
    }
    
    __bidevice__ 
        CoefficientSpectrum operator-(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] -= s2.c[i];
        }
        
        return ret;
    }
    
    __bidevice__ 
        CoefficientSpectrum &operator*=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] *= s2.c[i];
        }
        
        return *this;
    }
    
    __bidevice__ 
        CoefficientSpectrum operator*(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] *= s2.c[i];
        }
        
        return ret;
    }
    
    __bidevice__ 
        CoefficientSpectrum &operator/=(const CoefficientSpectrum &s2){
        for(int i = 0; i < nSpectrumSamples; i++){
            c[i] /= s2.c[i];
        }
        
        return *this;
    }
    
    __bidevice__ 
        CoefficientSpectrum operator/(const CoefficientSpectrum &s2) const{
        CoefficientSpectrum ret = *this;
        for(int i = 0; i < nSpectrumSamples; i++){
            ret.c[i] /= s2.c[i];
        }
        
        return ret;
    }
    
    __bidevice__ 
        CoefficientSpectrum operator*(float a) const { 
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] *= a;
        return ret;
    }
    
    __bidevice__ 
        CoefficientSpectrum &operator*=(float a) { 
        for (int i = 0; i < nSpectrumSamples; ++i)
            c[i] *= a;
        return *this; 
    }
    
    __bidevice__ 
        friend inline CoefficientSpectrum operator*(float a,const CoefficientSpectrum &s)
    {
        return s * a;
    }
    
    __bidevice__ 
        CoefficientSpectrum operator/(float a) const { 
        CoefficientSpectrum ret = *this;
        for (int i = 0; i < nSpectrumSamples; ++i){
            ret.c[i] /= a;
        }
        
        return ret;
    }
    
    __bidevice__ 
        CoefficientSpectrum &operator/=(float a) { 
        for (int i = 0; i < nSpectrumSamples; ++i){
            c[i] /= a;
        }
        
        return *this;
    }
    
    __bidevice__ 
        bool operator==(const CoefficientSpectrum &sp) const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != sp.c[i]) return false;
        return true;
    }
    
    __bidevice__ 
        bool operator!=(const CoefficientSpectrum &sp) const {
        return !(*this == sp);
    }
    
    __bidevice__ 
        bool IsBlack() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != 0.) return false;
        return true;
    }
    
    __bidevice__ 
        friend CoefficientSpectrum Sqrt(const CoefficientSpectrum &s) { 
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = glm::sqrt(s.c[i]);
        return ret;
    }
    
    template <int n> __bidevice__  friend inline CoefficientSpectrum<n> Pow(const CoefficientSpectrum<n> &s, float e);
    
    __bidevice__ 
        CoefficientSpectrum operator-() const {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = -c[i];
        return ret;
    }
    
    __bidevice__ 
        friend CoefficientSpectrum Exp(const CoefficientSpectrum &s) {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = glm::exp(s.c[i]);
        return ret;
    }
    
    __bidevice__ 
        CoefficientSpectrum Clamp(float low = 0, float high = FLT_MAX) const {
        CoefficientSpectrum ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = clamp(c[i], low, high);
        return ret;
    }
    
    __bidevice__ 
        bool HasNaNs() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != c[i]) return true;
        return false;
    }
    
    __bidevice__ 
        float &operator[](int i) {
        return c[i];
    }
    
    __bidevice__ 
        float operator[](int i) const {
        return c[i];
    }
};


class RGBSpectrum : public CoefficientSpectrum<3>{
    public:
    __bidevice__ RGBSpectrum(float v = 0.f) : CoefficientSpectrum<3>(v) { }
    
    __bidevice__ RGBSpectrum(const CoefficientSpectrum<3> &v) :
    CoefficientSpectrum<3>(v) { }
    
    __bidevice__ RGBSpectrum(const RGBSpectrum &s){
        *this = s;
    }
    
    __bidevice__ static RGBSpectrum FromRGB(glm::vec3 v){
        RGBSpectrum s;
        s.c[0] = v.x;
        s.c[1] = v.y;
        s.c[2] = v.z;
        return s;
    }
    
    __bidevice__ static RGBSpectrum FromRGB(float r, float g, float b){
        RGBSpectrum s;
        s.c[0] = r;
        s.c[1] = g;
        s.c[2] = b;
        return s;
    }
    
    __bidevice__ static RGBSpectrum FromRGB(const float rgb[3],
                                            SpectrumType type =
                                            SpectrumType::Reflectance) 
    {
        RGBSpectrum s;
        s.c[0] = rgb[0];
        s.c[1] = rgb[1];
        s.c[2] = rgb[2];
        return s;
    }

    __bidevice__
        glm::vec3 ToRGB() const{
        return glm::vec3(c[0], c[1], c[2]);
    }
    
    __bidevice__ 
        void ToRGB(glm::vec3 *v) const {
        *v = glm::vec3(c[0], c[1], c[2]);
    }
    
    __bidevice__ 
        void ToRGB(float *rgb) const {
        rgb[0] = c[0];
        rgb[1] = c[1];
        rgb[2] = c[2];
    }
    
    __bidevice__ 
        const RGBSpectrum &ToRGBSpectrum() const {
        return *this;
    }
    
    __bidevice__ 
        void ToXYZ(float xyz[3]) const { 
        RGBToXYZ(c, xyz);
    }
    
    __bidevice__ 
        static RGBSpectrum FromXYZ(const float xyz[3],
                                   SpectrumType type = SpectrumType::Reflectance) {
        RGBSpectrum r;
        XYZToRGB(xyz, r.c);
        return r;
    }
    
    __bidevice__ 
        float y() const {
        const float YWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
        return YWeight[0] * c[0] + YWeight[1] * c[1] + YWeight[2] * c[2];
    }
};

typedef RGBSpectrum Spectrum;

#endif