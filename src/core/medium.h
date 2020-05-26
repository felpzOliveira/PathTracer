#pragma once

#include <geometry.h>

#define WARN_BOUNCE_COUNT 20
class MediumInteraction;

__bidevice__ Float PhaseHG(Float cosTheta, Float g);

class PhaseFunction{
    public:
    Float g;
    int is_initialized;
    __bidevice__ PhaseFunction(): g(-2), is_initialized(0){}
    __bidevice__ PhaseFunction(Float g);
    __bidevice__ void SetG(Float g);
    
    __bidevice__ Float p(const vec3f &wo, const vec3f &wi) const;
    
    __bidevice__ Float Sample_p(const vec3f &wo, vec3f *wi, const Point2f &u) const;
};

class Medium{
    public:
    const Spectrum sigma_a, sigma_s, sigma_t;
    const Float g;
    
    __bidevice__ Medium(const Spectrum &sigma_a, const Spectrum &sigma_s, Float g)
        : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_a + sigma_s), g(g){}
    
    __bidevice__ Spectrum Tr(const Ray &ray) const;
    __bidevice__ Spectrum Sample(const Ray &ray, const Point2f &u, 
                                 MediumInteraction *mi) const;
    
    __bidevice__ void PrintSelf() const{
        printf("Medium: " v3fA(sigma_a) " " v3fA(sigma_s) " " v3fA(sigma_t) " G = %g\n",
               v3aA(sigma_a), v3aA(sigma_s), v3aA(sigma_t), g);
    }
};

class MediumInterface{
    public:
    const Medium *inside, *outside;
    
    __bidevice__ MediumInterface() : inside(nullptr), outside(nullptr) {}
    
    __bidevice__ MediumInterface(const Medium *medium) : inside(medium), outside(medium) {}
    
    __bidevice__ MediumInterface(const Medium *inside, const Medium *outside)
        : inside(inside), outside(outside) {}
    
    __bidevice__ bool IsMediumTransition() const { return inside != outside; }
};