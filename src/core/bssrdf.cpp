#include <bssrdf.h>
#include <medium.h>
#include <primitive.h>
#include <sampling.h>

//NOTE: This code needs to be studied
__bidevice__ Float FresnelMoment1(Float eta){
    Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
    eta5 = eta4 * eta;
    if(eta < 1)
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
        2.49277f * eta4 - 0.68441f * eta5;
    else
        return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
        1.27198f * eta4 + 0.12746f * eta5;
}

__bidevice__ Float FresnelMoment2(Float eta){
    Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
    eta5 = eta4 * eta;
    if (eta < 1) {
        return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 +
            0.07883f * eta4 + 0.04860f * eta5;
    } else {
        Float r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
        return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 +
            458.843f * r_eta + 404.557f * eta - 189.519f * eta2 +
            54.9327f * eta3 - 9.00603f * eta4 + 0.63942f * eta5;
    }
}

__bidevice__ bool CatmullRomWeights(int size, const Float *nodes, Float x, 
                                    int *offset, Float *weights)
{
    // Return _false_ if _x_ is out of bounds
    if (!(x >= nodes[0] && x <= nodes[size - 1])) return false;
    
    // Search for the interval _idx_ containing _x_
    int idx = FindInterval(size, [&](int i) { return nodes[i] <= x; });
    *offset = idx - 1;
    Float x0 = nodes[idx], x1 = nodes[idx + 1];
    
    // Compute the $t$ parameter and powers
    Float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
    
    // Compute initial node weights $w_1$ and $w_2$
    weights[1] = 2 * t3 - 3 * t2 + 1;
    weights[2] = -2 * t3 + 3 * t2;
    
    // Compute first node weight $w_0$
    if (idx > 0) {
        Float w0 = (t3 - 2 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        Float w0 = t3 - 2 * t2 + t;
        weights[0] = 0;
        weights[1] -= w0;
        weights[2] += w0;
    }
    
    // Compute last node weight $w_3$
    if (idx + 2 < size) {
        Float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        Float w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0;
    }
    return true;
}

__bidevice__ Float IntegrateCatmullRom(int n, const Float *x, const Float *values, 
                                       Float *cdf)
{
    Float sum = 0;
    cdf[0] = 0;
    for (int i = 0; i < n - 1; ++i) {
        // Look up $x_i$ and function values of spline segment _i_
        Float x0 = x[i], x1 = x[i + 1];
        Float f0 = values[i], f1 = values[i + 1];
        Float width = x1 - x0;
        
        // Approximate derivatives using finite differences
        Float d0, d1;
        if (i > 0)
            d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
        else
            d0 = f1 - f0;
        if (i + 2 < n)
            d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
        else
            d1 = f1 - f0;
        
        // Keep a running sum and build a cumulative distribution function
        sum += ((d0 - d1) * (1.f / 12.f) + (f0 + f1) * .5f) * width;
        cdf[i + 1] = sum;
    }
    return sum;
}

__bidevice__ Float SampleCatmullRom2D(int size1, int size2, const Float *nodes1,
                                      const Float *nodes2, const Float *values,
                                      const Float *cdf, Float alpha, Float u, 
                                      Float *fval = nullptr, Float *pdf = nullptr)
{
    // Determine offset and coefficients for the _alpha_ parameter
    int offset;
    Float weights[4];
    if (!CatmullRomWeights(size1, nodes1, alpha, &offset, weights)) return 0;
    
    // Define a lambda function to interpolate table entries
    auto interpolate = [&](const Float *array, int idx) {
        Float value = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
            value += array[(offset + i) * size2 + idx] * weights[i];
        return value;
    };
    
    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    Float maximum = interpolate(cdf, size2 - 1);
    u *= maximum;
    int idx =
        FindInterval(size2, [&](int i) { return interpolate(cdf, i) <= u; });
    
    // Look up node positions and interpolated function values
    Float f0 = interpolate(values, idx), f1 = interpolate(values, idx + 1);
    Float x0 = nodes2[idx], x1 = nodes2[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;
    
    // Re-scale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;
    
    // Approximate derivatives using finite differences of the interpolant
    if (idx > 0)
        d0 = width * (f1 - interpolate(values, idx - 1)) /
        (x1 - nodes2[idx - 1]);
    else
        d0 = f1 - f0;
    if (idx + 2 < size2)
        d1 = width * (interpolate(values, idx + 2) - f0) /
        (nodes2[idx + 2] - x0);
    else
        d1 = f1 - f0;
    
    // Invert definite integral over spline segment and return solution
    
    // Set initial guess for $t$ by importance sampling a linear interpolant
    Float t;
    if (f0 != f1)
        t = (f0 - std::sqrt(Max((Float)0, f0 * f0 + 2 * u * (f1 - f0)))) / (f0 - f1);
    else
        t = u / f0;
    Float a = 0, b = 1, Fhat, fhat;
    while (true) {
        // Fall back to a bisection step when _t_ is out of bounds
        if (!(t >= a && t <= b)) t = 0.5f * (a + b);
        
        // Evaluate target function and its derivative in Horner form
        Fhat = t * (f0 +
                    t * (.5f * d0 +
                         t * ((1.f / 3.f) * (-2 * d0 - d1) + f1 - f0 +
                              t * (.25f * (d0 + d1) + .5f * (f0 - f1)))));
        fhat = f0 +
            t * (d0 +
                 t * (-2 * d0 - d1 + 3 * (f1 - f0) +
                      t * (d0 + d1 + 2 * (f0 - f1))));
        
        // Stop the iteration if converged
        if (Absf(Fhat - u) < 1e-6f || b - a < 1e-6f) break;
        
        // Update bisection bounds using updated _t_
        if (Fhat - u < 0)
            a = t;
        else
            b = t;
        
        // Perform a Newton step
        t -= (Fhat - u) / fhat;
    }
    
    // Return the sample position and function value
    if (fval) *fval = fhat;
    if (pdf) *pdf = fhat / maximum;
    return x0 + width * t;
}

__bidevice__ Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r){
    const int nSamples = 100;
    Float Ed = 0;
    // Precompute information for dipole integrand
    
    // Compute reduced scattering coefficients $\sigmaps, \sigmapt$ and albedo
    // $\rhop$
    Float sigmap_s = sigma_s * (1 - g);
    Float sigmap_t = sigma_a + sigmap_s;
    Float rhop = sigmap_s / sigmap_t;
    
    // Compute non-classical diffusion coefficient $D_\roman{G}$ using
    // Equation (15.24)
    Float D_g = (2 * sigma_a + sigmap_s) / (3 * sigmap_t * sigmap_t);
    
    // Compute effective transport coefficient $\sigmatr$ based on $D_\roman{G}$
    Float sigma_tr = std::sqrt(sigma_a / D_g);
    
    // Determine linear extrapolation distance $\depthextrapolation$ using
    // Equation (15.28)
    Float fm1 = FresnelMoment1(eta), fm2 = FresnelMoment2(eta);
    Float ze = -2 * D_g * (1 + 3 * fm2) / (1 - 2 * fm1);
    
    // Determine exitance scale factors using Equations (15.31) and (15.32)
    Float cPhi = .25f * (1 - 2 * fm1), cE = .5f * (1 - 3 * fm2);
    for (int i = 0; i < nSamples; ++i) {
        // Sample real point source depth $\depthreal$
        Float zr = -std::log(1 - (i + .5f) / nSamples) / sigmap_t;
        
        // Evaluate dipole integrand $E_{\roman{d}}$ at $\depthreal$ and add to
        // _Ed_
        Float zv = -zr + 2 * ze;
        Float dr = std::sqrt(r * r + zr * zr), dv = std::sqrt(r * r + zv * zv);
        
        // Compute dipole fluence rate $\dipole(r)$ using Equation (15.27)
        Float phiD = Inv4Pi / D_g * (std::exp(-sigma_tr * dr) / dr -
                                     std::exp(-sigma_tr * dv) / dv);
        
        // Compute dipole vector irradiance $-\N{}\cdot\dipoleE(r)$ using
        // Equation (15.27)
        Float EDn = Inv4Pi * (zr * (1 + sigma_tr * dr) *
                              std::exp(-sigma_tr * dr) / (dr * dr * dr) -
                              zv * (1 + sigma_tr * dv) *
                              std::exp(-sigma_tr * dv) / (dv * dv * dv));
        
        // Add contribution from dipole for depth $\depthreal$ to _Ed_
        Float E = phiD * cPhi + EDn * cE;
        Float kappa = 1 - std::exp(-2 * sigmap_t * (dr + zr));
        Ed += kappa * rhop * rhop * E;
    }
    return Ed / nSamples;
}

__bidevice__ Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r){
    // Compute material parameters and minimum $t$ below the critical angle
    Float sigma_t = sigma_a + sigma_s, rho = sigma_s / sigma_t;
    Float tCrit = r * std::sqrt(eta * eta - 1);
    Float Ess = 0;
    const int nSamples = 100;
    for (int i = 0; i < nSamples; ++i) {
        // Evaluate single scattering integrand and add to _Ess_
        Float ti = tCrit - std::log(1 - (i + .5f) / nSamples) / sigma_t;
        
        // Determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        Float d = std::sqrt(r * r + ti * ti);
        Float cosThetaO = ti / d;
        
        // Add contribution of single scattering at depth $t$
        Ess += rho * std::exp(-sigma_t * (d + tCrit)) / (d * d) *
            PhaseHG(cosThetaO, g) * (1 - FrDieletric(-cosThetaO, 1, eta)) *
            std::abs(cosThetaO);
    }
    return Ess / nSamples;
}

__bidevice__ void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t){
    // Choose radius values of the diffusion profile discretization
    t->radiusSamples[0] = 0;
    t->radiusSamples[1] = 2.5e-3f;
    for(int i = 2; i < t->nRadiusSamples; ++i)
        t->radiusSamples[i] = t->radiusSamples[i - 1] * 1.2f;
    
    // Choose albedo values of the diffusion profile discretization
    for(int i = 0; i < t->nRhoSamples; ++i){
        t->rhoSamples[i] = (1 - std::exp(-8 * i / (Float)(t->nRhoSamples - 1))) /
            (1 - std::exp(-8.f));
    }
    
    for(int i = 0; i < t->nRhoSamples; i++){
        // Compute the diffusion profile for the _i_th albedo sample
        // Compute scattering profile for chosen albedo $\rho$
        for (int j = 0; j < t->nRadiusSamples; ++j) {
            Float rho = t->rhoSamples[i], r = t->radiusSamples[j];
            t->profile[i * t->nRadiusSamples + j] =
                2 * Pi * r * (BeamDiffusionSS(rho, 1 - rho, g, eta, r) +
                              BeamDiffusionMS(rho, 1 - rho, g, eta, r));
        }
        
        // Compute effective albedo $\rho_{\roman{eff}}$ and CDF for importance
        // sampling
        t->rhoEff[i] =
            IntegrateCatmullRom(t->nRadiusSamples, t->radiusSamples,
                                &t->profile[i * t->nRadiusSamples],
                                &t->profileCDF[i * t->nRadiusSamples]);
    }
}

__bidevice__ BSSRDFTable::BSSRDFTable(int nRhoSamples, int nRadiusSamples)
: nRhoSamples(nRhoSamples), nRadiusSamples(nRadiusSamples)
{
    rhoSamples = new Float[nRhoSamples];
    radiusSamples = new Float[nRadiusSamples];
    profile = new Float[nRadiusSamples * nRhoSamples];
    rhoEff = new Float[nRhoSamples];
    profileCDF = new Float[nRadiusSamples * nRhoSamples];
}

///////////////////////////////////////////////////////////////////////////////////////

__bidevice__ Spectrum SeparableBSSRDF::Tabulated_Sr(Float r) const{
    Spectrum Sr(0.f);
    int specChannels = 3;
    for(int ch = 0; ch < specChannels; ++ch){
        Float rOptical = r * sigma_t[ch];
        int rhoOffset, radiusOffset;
        Float rhoWeights[4], radiusWeights[4];
        if (!CatmullRomWeights(table->nRhoSamples, table->rhoSamples,
                               rho[ch], &rhoOffset, rhoWeights) ||
            !CatmullRomWeights(table->nRadiusSamples, table->radiusSamples,
                               rOptical, &radiusOffset, radiusWeights))
            continue;
        
        Float sr = 0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Float weight = rhoWeights[i] * radiusWeights[j];
                if (!IsZero(weight))
                    sr += weight * table->EvalProfile(rhoOffset + i, radiusOffset + j);
            }
        }
        
        if (!IsZero(rOptical)) sr /= 2 * Pi * rOptical;
        Sr[ch] = sr;
    }
    
    Sr *= sigma_t * sigma_t;
    return Clamp(Sr);
}

__bidevice__ Float SeparableBSSRDF::Tabulated_Pdf_Sr(int ch, Float r) const{
    Float rOptical = r * sigma_t[ch];
    int rhoOffset, radiusOffset;
    
    Float rhoWeights[4], radiusWeights[4];
    if (!CatmullRomWeights(table->nRhoSamples, table->rhoSamples, rho[ch],
                           &rhoOffset, rhoWeights) ||
        !CatmullRomWeights(table->nRadiusSamples, table->radiusSamples,
                           rOptical, &radiusOffset, radiusWeights))
        return 0.f;
    
    Float sr = 0, rhoEff = 0;
    for (int i = 0; i < 4; ++i) {
        if (IsZero(rhoWeights[i])) continue;
        rhoEff += table->rhoEff[rhoOffset + i] * rhoWeights[i];
        for (int j = 0; j < 4; ++j) {
            if (IsZero(radiusWeights[j])) continue;
            sr += table->EvalProfile(rhoOffset + i, radiusOffset + j) *
                rhoWeights[i] * radiusWeights[j];
        }
    }
    
    if (!IsZero(rOptical)) sr /= 2 * Pi * rOptical;
    return Max((Float)0, sr * sigma_t[ch] * sigma_t[ch] / rhoEff);
}

__bidevice__ Float SeparableBSSRDF::Tabulated_Sample_Sr(int ch, Float u) const{
    if (IsZero(sigma_t[ch])) return -1;
    return SampleCatmullRom2D(table->nRhoSamples, table->nRadiusSamples,
                              table->rhoSamples, table->radiusSamples,
                              table->profile, table->profileCDF,
                              rho[ch], u) / sigma_t[ch];
}

__bidevice__ Spectrum SeparableBSSRDF::Sample_S(Aggregator *scene, BSDF *bsdf, Float u1, 
                                                const Point2f &u2, SurfaceInteraction *si, 
                                                Float *pdf, SurfaceInteraction *isect)
{
    Spectrum Sp = Sample_Sp(scene, u1, u2, si, pdf, isect);
    if(!Sp.IsBlack()){
        BxDF bxdf(BxDFImpl::BSSRDFAdapter);
        bxdf.Init_BSSRDF(this);
        bsdf->Push(&bxdf);
        si->wo = ToVec3(si->n);
    }
    
    return Sp;
}

__bidevice__ Spectrum SeparableBSSRDF::Sample_Sp(Aggregator *scene, Float u1, 
                                                 const Point2f &u2, SurfaceInteraction *pi, 
                                                 Float *pdf, SurfaceInteraction *isect) const
{
    vec3f vx, vy, vz;
    int specChannels = 3;
    if(u1 < .5f){
        vx = ss; vy = ts; vz = ToVec3(ns); u1 *= 2;
    }else if(u1 < .75f){
        // Prepare for sampling rays with respect to _ss_
        vx = ts; vy = ToVec3(ns); vz = ss;
        u1 = (u1 - .5f) * 4;
    }else{
        // Prepare for sampling rays with respect to _ts_
        vx = ToVec3(ns); vy = ss; vz = ts;
        u1 = (u1 - .75f) * 4;
    }
    
    int ch = Clamp((int)(u1 * specChannels), 0, specChannels - 1);
    u1 = u1 * specChannels - ch;
    Float r = Sample_Sr(ch, u2[0]);
    
    if(r < 0) return Spectrum(0.f);
    Float phi = 2 * Pi * u2[1];
    
    Float rMax = Sample_Sr(ch, 0.999f);
    if(r >= rMax) return Spectrum(0.f);
    Float l = 2 * std::sqrt(rMax * rMax - r * r);
    
    Interaction base;
    base.p = po.p + r * (vx * std::cos(phi) + vy * std::sin(phi)) - l * vz * 0.5f;
    base.time = po.time;
    Point3f pTarget = base.p + l * vz;
    
    /*
    * Alright look, I know this should be a dynamic list. But we cannot aford to 
    * do that, so here is the deal: after running a few samples of f11-15 they only
    * sample the material a couple of times [ 0 ~ 2 ], because I have no idea how many
    * interactions we are going to need I'm gonna let it with a fixed interactions. This
    * might be bad, but until we have a image this is whats happening.
*/
#define MAX_SS_SI 256
    SurfaceInteraction si[MAX_SS_SI];
    int maxSi = MAX_SS_SI;
#undef MAX_SS_SI
    
    /*
    * Hi, its me again, the writer of this mess. I have a problem with the following
    * loop. I understand its meant to ignore close geometry. But when geometry is *too*
    * close this thing simply never ends, and that condition for adding the material
    * is just too sketchy, like I can't have 2 meshes using the same material?
    * I also don't understand why do we need to consider the whole scene, can't we
    * simply intersect the original primitive ? I'm putting this here
    * as it is a huge speedup and I don't see any visual changes
*/
    int nFound = 0;
    SurfaceInteraction *ptr = &si[nFound];
    while(true){
        Ray ray = base.SpawnRayTo(pTarget);
        bool hit = PrimitiveIntersect(isect->primitive, ray, ptr);
        if(ray.d.IsBlack() || !hit) break;
        
        base = *ptr;
        if(ptr->primitive == isect->primitive){
            nFound++;
            ptr = &si[nFound];
            if(nFound > maxSi-1){
                //Float dist = Distance(pTarget, ptr->p);
                //printf("Warning: Not enough interaction vectors for sampling BSSRDF [%g]\n", dist);
                break;
            }
        }
    }
    
    if(nFound == 0) return Spectrum(0.0f);
    int selected = Clamp((int)(u1 * nFound), 0, nFound - 1);
    *pi = si[selected];
    *pdf = this->Pdf_Sp(pi) / nFound;
    return this->Sp(*pi);
}

__bidevice__ Float SeparableBSSRDF::Pdf_Sp(SurfaceInteraction *pi) const{
    int specChannels = 3;
    vec3f d = po.p - pi->p;
    vec3f dLocal(Dot(ss, d), Dot(ts, d), Dot(ns, d));
    Normal3f nLocal(Dot(ss, ToVec3(pi->n)), Dot(ts, ToVec3(pi->n)), Dot(ns, ToVec3(pi->n)));
    
    Float rProj[3] = {std::sqrt(dLocal.y * dLocal.y + dLocal.z * dLocal.z),
        std::sqrt(dLocal.z * dLocal.z + dLocal.x * dLocal.x),
        std::sqrt(dLocal.x * dLocal.x + dLocal.y * dLocal.y)};
    
    Float pdf = 0, axisProb[3] = {.25f, .25f, .5f};
    Float chProb = 1 / (Float)specChannels;
    for(int axis = 0; axis < 3; ++axis){
        for(int ch = 0; ch < specChannels; ++ch){
            pdf += Pdf_Sr(ch, rProj[axis]) * Absf(nLocal[axis]) * chProb * axisProb[axis];
        }
    }
    
    return pdf;
}

// Overall schlick approximation of the fresnel term
inline __bidevice__ Float SchlickWeight(Float cosTheta){
    Float m = Clamp(1 - cosTheta, 0, 1);
    return (m * m) * (m * m) * m;
}

inline __bidevice__ Float FrSchlick(Float R0, Float cosTheta) {
    return Lerp(SchlickWeight(cosTheta), R0, 1.f);
}

inline __bidevice__ Spectrum FrSchlick(const Spectrum &R0, Float cosTheta) {
    return Lerp(SchlickWeight(cosTheta), R0, Spectrum(1.));
}

__bidevice__ Spectrum SeparableBSSRDF::Disney_S(const SurfaceInteraction &pi, const vec3f &wi){
    vec3f a = Normalize(pi.p - po.p);
    Float fade = 1;
    vec3f n = ToVec3(po.n);
    Float cosTheta = Dot(a, n);
    if(cosTheta > 0){
        Float sinTheta = std::sqrt(Max(Float(0), 1 - cosTheta * cosTheta));
        vec3f a2 = n * sinTheta - (a - n * cosTheta) * cosTheta / sinTheta;
        fade = Max(Float(0), Dot(pi.n, a2));
    }
    
    Float Fo = SchlickWeight(AbsCosTheta(po.wo));
    Float Fi = SchlickWeight(AbsCosTheta(wi));
    // this is the relation from page 6 ( relation 4 ). The fade term
    // is comming from PBRT, one of those things that if you are not
    // in direct contact with people who publish it cause this isn't in
    // the work.
    return fade * (1 - Fo / 2) * (1 - Fi / 2) * Sp(pi) / Pi;
}

__bidevice__ Spectrum SeparableBSSRDF::Disney_Sr(Float r) const{
    if(r < 1e-6f) r = 1e-6f; //zero term
    // the following is the direct computation of the subsurface diffusion
    // in page 7 ( relation 5 ) of the original work.
    return R * (Exp(-Spectrum(r) / d) + Exp(-Spectrum(r) / (3 * d))) /
        (8 * Pi * d * r);
}

__bidevice__ Float SeparableBSSRDF::Disney_Sample_Sr(int ch, Float u) const{
    /*
    * PBRT:
    * Because Sr is normalized the CDF for that is:
    * 1 - e^(-x/d) / 4 - (3 / 4) e^(-x / (3d))
    * Split the sampling into 2 terms:
    *  - e^(-r/d) / (2 Pi d r), CDF: 1 - e^(-r/d), sampling: r = d log(1 / (1 - u))
    *  - e^(-r/(3d)) / (6 Pi d r), CDF: 1 - e^(-r/(3d)), sampling: r = 3 d log(1 / (1 - u))
    */
    
    if(u < 0.25f){
        u = Min(u * 4, OneMinusEpsilon); //normalize
        return d[ch] * std::log(1 / (1 - u));
    }else{
        u = Min((u - .25f) / .75f, OneMinusEpsilon); //normalize
        return 3 * d[ch] * std::log(1 / (1 - u));
    }
}


__bidevice__ Spectrum SeparableBSSRDF::Tabulated_S(const SurfaceInteraction &pi, const vec3f &wi){
    Float Ft = FrDieletric(CosTheta(po.wo), 1, eta);
    return (1 - Ft) * Sp(pi) * Sw(wi);
}

__bidevice__ Float SeparableBSSRDF::Pdf_Sr(int ch, Float r) const{
    switch(type){
        case BSSRDFType::BSSRDFTabulated:{
            return Tabulated_Pdf_Sr(ch, r);
        } break;
        
        case BSSRDFType::BSSRDFDisney:{
            return Disney_Pdf_Sr(ch, r);
        } break;
        
        default:{
            printf("Unknown BSSRDF implementation\n");
            return 0;
        }
    }
}

__bidevice__ Float SeparableBSSRDF::Sample_Sr(int ch, Float u) const{
    switch(type){
        case BSSRDFType::BSSRDFTabulated:{
            return Tabulated_Sample_Sr(ch, u);
        } break;
        
        case BSSRDFType::BSSRDFDisney:{
            return Disney_Sample_Sr(ch, u);
        } break;
        
        default:{
            printf("Unknown BSSRDF implementation\n");
            return 0;
        }
    }
}

__bidevice__ Spectrum SeparableBSSRDF::Sr(Float d) const{
    switch(type){
        case BSSRDFType::BSSRDFTabulated:{
            return Tabulated_Sr(d);
        } break;
        
        case BSSRDFType::BSSRDFDisney:{
            return Disney_Sr(d);
        } break;
        
        default:{
            printf("Unknown BSSRDF implementation\n");
            return Spectrum(0);
        }
    }
}

__bidevice__ Spectrum SeparableBSSRDF::S(const SurfaceInteraction &pi, const vec3f &wi){
    switch(type){
        case BSSRDFType::BSSRDFTabulated:{
            return Tabulated_S(pi, wi);
        } break;
        
        case BSSRDFType::BSSRDFDisney:{
            return Disney_S(pi, wi);
        } break;
        
        default:{
            printf("Unknown BSSRDF implementation\n");
            return Spectrum(0);
        }
    }
}

__bidevice__ Float SeparableBSSRDF::Disney_Pdf_Sr(int ch, Float r) const{
    if (r < 1e-6f) r = 1e-6f; //zero term
    // weighted
    return (.25f * std::exp(-r / d[ch]) / (2 * Pi * d[ch] * r) +
            .75f * std::exp(-r / (3 * d[ch])) / (6 * Pi * d[ch] * r));
}


__bidevice__ void SeparableBSSRDF::Init_TabulatedBSSRDF(const Spectrum &sigma_a, 
                                                        const Spectrum &sigma_s, 
                                                        BSSRDFTable *t)
{
    table = t;
    type = BSSRDFType::BSSRDFTabulated;
    sigma_t = sigma_a + sigma_s;
    //printf(v3fA(sigma_t) " " v3fA(sigma_a) " " v3fA(sigma_s) "\n", v3aA(sigma_t), v3aA(sigma_a), v3aA(sigma_s));
    for(int c = 0; c < 3; ++c)
        rho[c] = !IsZero(sigma_t[c]) ? (sigma_s[c] / sigma_t[c]) : 0;
}

__bidevice__ void SeparableBSSRDF::Init_DisneyBSSRDF(const Spectrum &r, const Spectrum &s){
    R = r;
    d = 0.2 * s;
    type = BSSRDFType::BSSRDFDisney;
}
