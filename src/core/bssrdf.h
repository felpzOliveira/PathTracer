#pragma once

#include <interaction.h>

class Material;
class Aggregator;
class BSDF;

__bidevice__ Float FrDieletric(Float cosThetaI, Float etaI, Float etaT);

/*
* Update 1 : I'm about to attempt to implement this, save yourself, going to read about it.
* Update 2 : Finished reading. Alright ... this is hard as hell. I tot this was 
 * going to be something like 'ok, its easy... BSSRDF is just like BSDF * 2'. 
* But no this is a whole new beast. You litteraly need to be a god to simply uderstand 
* this, equations are hard, the approximation (1-Fr)SwSp helps but still... 
* I don't even want to know how am I supposed to write cuda compatible code from the 
* pbrt implementation. I'm going to start with the Disney implementation as it is 
* math only and don't rely on chaining, and we'll see where we go from there. 
* I really want those spheres in figure 11.15... god dammit Q_Q. Please be gentle Disney.
*/

/*
 * Computes normalization therms on the splitted S computation, per relation 11.8
*/
__bidevice__ Float FresnelMoment1(Float invEta);
__bidevice__ Float FresnelMoment2(Float invEta);

struct BSSRDFTable{
    // BSSRDFTable Public Data
    const int nRhoSamples, nRadiusSamples;
    Float *rhoSamples, *radiusSamples;
    Float *profile;
    Float *rhoEff;
    Float *profileCDF;
    
    // BSSRDFTable Public Methods
    __bidevice__ BSSRDFTable(int nRhoSamples, int nRadiusSamples);
    inline __bidevice__ Float EvalProfile(int rhoIndex, int radiusIndex) const{
        return profile[rhoIndex * nRadiusSamples + radiusIndex];
    }
};

/*
* This is the generic interface, it is supposed to represent the eight dimensional
* equation on 11.14. However this is a beast of equation, so we'll not really solve
* it, we'll split that equation into three factors, i.e.:
*      A fresnel factor ->  1 - Fr(cos(theta));
*      A [second] fresnel factor -> Sw(wi);
*      A spatial factor -> Sp(p0,pi);
* The splitted equation relies on the relation 11.6 which should be implemented
* in the class SeparableBSSRDF. The class (not here yet) TabulatedBSSRDF presents
* this notion with a table of profiles, I don't want to implement that because
* it relies on dynamic lists for intersection testing which cuda will cry about,
* hopefully Disney one doesnt.
*/
#if 0
class BSSRDF{
    public:
    const SurfaceInteraction &po;
    Float eta;
    
    __bidevice__ BSSRDF(const SurfaceInteraction &po, Float eta) : po(po), eta(eta) {}
    __bidevice__ virtual Spectrum S(const SurfaceInteraction &pi, const vec3f &wi) = 0;
    __bidevice__ virtual Spectrum Sample_S(Aggregator *scene, BSDF *bsdf, Float u1, 
                                           const Point2f &u2, SurfaceInteraction *si, 
                                           Float *pdf) = 0;
};
#endif

enum BSSRDFType{
    BSSRDFTabulated, BSSRDFDisney, BSSRDFNone
};

/*
* Split S(p0,w0,pi,wi) ~ (1 - Fr(cos(theta))) Sp(p0,pi) Sw(wi)
* For Sw(wi) we use the approximation in relation 11.7, and use 11.8 to compute
* normalization with the Fresnel moment. For Sp(p0,pi) we assume that this function
* is a function of distance and therefore Sp(p0,pi) ~ Sp(|p0-pi|) and therefore we 
* write that Sp(p0,pi) ~ Sp(|p0-pi|) = Sr(|p0-pi|) where Sr is some distribution 
* that captures the process for the distance |p0-pi|. Classes that implement this
* should provide the distribution Sr and the Pdf relation to that distribution.
*/
class SeparableBSSRDF{
    public:
    SurfaceInteraction po;
    Float eta;
    Normal3f ns;
    vec3f ss, ts;
    Material *material;
    TransportMode mode;
    BSSRDFType type;
    
    // For tabulated
    BSSRDFTable *table;
    Spectrum sigma_t, rho;
    
    // For disney
    Spectrum R, d;
    
    
    __bidevice__ SeparableBSSRDF(){}
    __bidevice__ SeparableBSSRDF(const SurfaceInteraction &po, Float eta,
                                 Material *material, TransportMode mode)
        : po(po), eta(eta), ns(po.n), ss(Normalize(po.dpdu)), ts(Cross(ToVec3(ns), ss)),
    material(material), mode(mode) { type = BSSRDFNone; }
    
    __bidevice__ void Initialize(const SurfaceInteraction &ppo, Float e,
                                 Material *mat, TransportMode md)
    {
        po = ppo; eta = e; ns = po.n;
        ss = Normalize(po.dpdu);
        ts = Cross(ToVec3(ns), ss);
        material = mat;
        mode = md;
        type = BSSRDFNone;
    }
    
    __bidevice__ void Init_TabulatedBSSRDF(const Spectrum &sigma_a, const Spectrum &sigma_s, 
                                           BSSRDFTable *table);
    
    __bidevice__ void Init_DisneyBSSRDF(const Spectrum &R, const Spectrum &d);
    
    __bidevice__ Spectrum S(const SurfaceInteraction &pi, const vec3f &wi);
    
    __bidevice__ Spectrum Sw(const vec3f &w) const{
        Float c = 1 - 2 * FresnelMoment1(1 / eta);
        return (1 - FrDieletric(CosTheta(w), 1, eta)) / (c * Pi);
    }
    
    __bidevice__ Spectrum Sp(const SurfaceInteraction &pi) const{
        return Sr(Distance(po.p, pi.p));
    }
    
    __bidevice__ Float Pdf_Sp(SurfaceInteraction *pi) const;
    
    __bidevice__ Spectrum Sample_Sp(Aggregator *scene, Float u1, const Point2f &u2,
                                    SurfaceInteraction *si, Float *pdf,
                                    SurfaceInteraction *isect) const;
    
    __bidevice__ virtual Spectrum Sample_S(Aggregator *scene, BSDF *bsdf, Float u1, 
                                           const Point2f &u2, SurfaceInteraction *si, 
                                           Float *pdf, SurfaceInteraction *isect);
    
    __bidevice__ Spectrum Sr(Float d) const;
    __bidevice__ Float Sample_Sr(int ch, Float u) const;
    __bidevice__ Float Pdf_Sr(int ch, Float r) const;
    
    
    private:
    __bidevice__ Spectrum Disney_S(const SurfaceInteraction &pi, const vec3f &wi);
    __bidevice__ Spectrum Disney_Sr(Float d) const;
    __bidevice__ Float Disney_Sample_Sr(int ch, Float u) const;
    __bidevice__ Float Disney_Pdf_Sr(int ch, Float r) const;
    
    __bidevice__ Spectrum Tabulated_S(const SurfaceInteraction &pi, const vec3f &wi);
    __bidevice__ Spectrum Tabulated_Sr(Float d) const;
    __bidevice__ Float Tabulated_Sample_Sr(int ch, Float u) const;
    __bidevice__ Float Tabulated_Pdf_Sr(int ch, Float r) const;
};

__bidevice__ Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r);
__bidevice__ Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r);
__bidevice__ void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t);
__bidevice__ void SubsurfaceFromDiffuse(const BSSRDFTable &table, const Spectrum &rhoEff,
                                        const Spectrum &mfp, Spectrum *sigma_a,
                                        Spectrum *sigma_s);