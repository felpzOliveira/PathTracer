#include <medium.h>
#include <interaction.h>

__bidevice__ Float PhaseHG(Float cosTheta, Float g){
    Float denom = 1 + g * g + 2 * g * cosTheta;
    if(IsZero(denom)){
        /*
        * Look, 1+x^2+2xa is not really 0 anywhere for a in [-1,1]
        * However miss initialization of g can cause this function to zero
        * and than things will crash because we prevent any miss-computation
        * in this source code, so I'm adding a print to make sure whatever 
        * medium called this has correct parameters.
*/
        printf("Zero evaluation: Denom: %g, G: %g, cosTheta: %g\n", denom, g, cosTheta);
    }
    
    AssertAEx(!IsZero(denom), "Zero denominator on Phase computation");
    return Inv4Pi * (1 - g * g) / (denom * std::sqrt(denom));
}


__bidevice__ PhaseFunction::PhaseFunction(Float g) : g(g){is_initialized = 1;}

__bidevice__ void PhaseFunction::SetG(Float _g){g = _g; is_initialized = 1;}

__bidevice__ Float PhaseFunction::p(const vec3f &wo, const vec3f &wi) const{
    AssertAEx(is_initialized == 1, "Invalid call to PhaseFunction::p");
    return PhaseHG(Dot(wo, wi), g);
}

__bidevice__ Float PhaseFunction::Sample_p(const vec3f &wo, vec3f *wi, 
                                           const Point2f &u) const
{
    AssertA(is_initialized == 1, "Invalid call to PhaseFunction::Sample_p");
    Float cosTheta;
    if(Absf(g) < 1e-3){
        cosTheta = 1 - 2 * u[0];
    }else{
        Float sqrTerm = (1 - g * g) / (1 - g + 2 * g * u[0]);
        cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
    }
    
    Float sinTheta = std::sqrt(Max((Float)0, 1 - cosTheta * cosTheta));
    Float phi = 2 * Pi * u[1];
    vec3f v1, v2;
    CoordinateSystem(wo, &v1, &v2);
    *wi = SphericalDirection(sinTheta, cosTheta, phi, v1, v2, -wo);
    return PhaseHG(-cosTheta, g);
}

__bidevice__ Spectrum Medium::Tr(const Ray &ray) const{
    return Exp(-sigma_t * Min(ray.tMax * ray.d.Length(), MAX_FLT));
}

__bidevice__ Spectrum Medium::Sample(const Ray &ray, const Point2f &u, 
                                     MediumInteraction *mi) const
{
    int nn = 3;
    int channel = Min((int)(u[0] * nn), nn - 1);
    
    AssertAEx(!IsZero(sigma_t[channel]), "Zero sigma_t channel");
    
    Float dist = -std::log(1 - u[1]) / sigma_t[channel];
    Float len = ray.d.Length();
    
    AssertAEx(!IsZero(len), "Zero length direction");
    
    Float t = Min(dist / len, ray.tMax);
    bool sampledMedium = t < ray.tMax;
    
    if(sampledMedium){
        *mi = MediumInteraction(ray(t), -ray.d, ray.time, this, this->g);
    }
    
    Spectrum Tr = Exp(-sigma_t * Min(t, MAX_FLT) * ray.d.Length());
    Spectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    Float pdf = 0;
    for (int i = 0; i < nn; ++i) pdf += density[i];
    pdf *= 1 / (Float)nn;
    if(IsZero(pdf)){
        pdf = 1;
    }
    
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / pdf);
}