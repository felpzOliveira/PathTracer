#include <shape.h>

__bidevice__ Bounds3f Sphere::GetBounds() const{
    return (ObjectToWorld)(Bounds3f(Point3f(-radius, -radius, zMin),
                                    Point3f(radius, radius, zMax)));
}

__bidevice__ bool Sphere::Intersect(const Ray &r, Float *tHit, 
                                    SurfaceInteraction *isect) const
{
    Float phi;
    Point3f pHit;
    vec3f oErr, dErr;
    Ray ray = (WorldToObject)(r, &oErr, &dErr);
    
    Float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
    Float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;
    Float a = dx * dx + dy * dy + dz * dz;
    Float b = 2 * (dx * ox + dy * oy + dz * oz);
    Float c = ox * ox + oy * oy + oz * oz - Float(radius) * Float(radius);
    
    Float t0, t1;
    if(!Quadratic(a, b, c, &t0, &t1)) return false;
    
    if(t0 > ray.tMax || t1 <= 0) return false;
    Float tShapeHit = t0;
    if(tShapeHit <= 0){
        tShapeHit = t1;
        if(tShapeHit > ray.tMax) return false;
    }
    
    pHit = ray((Float)tShapeHit);
    
    pHit *= radius / Distance(pHit, Point3f(0, 0, 0));
    if (pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * radius;
    phi = std::atan2(pHit.y, pHit.x);
    if (phi < 0) phi += 2 * Pi;
    
    if((zMin > -radius && pHit.z < zMin) || (zMax < radius && pHit.z > zMax) || phi > phiMax)
    {
        if (tShapeHit == t1) return false;
        if (t1 > ray.tMax) return false;
        tShapeHit = t1;
        pHit = ray((Float)tShapeHit);
        
        pHit *= radius / Distance(pHit, Point3f(0, 0, 0));
        if(pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * radius;
        phi = std::atan2(pHit.y, pHit.x);
        if(phi < 0) phi += 2 * Pi;
        if((zMin > -radius && pHit.z < zMin) ||
           (zMax < radius && pHit.z > zMax) || phi > phiMax)
            return false;
    }
    
    Float u = phi / phiMax;
    Float theta = std::acos(Clamp(pHit.z / radius, -1, 1));
    Float v = (theta - thetaMin) / (thetaMax - thetaMin);
    
    Float zRadius = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
    Float invZRadius = 1 / zRadius;
    Float cosPhi = pHit.x * invZRadius;
    Float sinPhi = pHit.y * invZRadius;
    vec3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
    vec3f dpdv =
        (thetaMax - thetaMin) *
        vec3f(pHit.z * cosPhi, pHit.z * sinPhi, -radius * std::sin(theta));
    
    vec3f d2Pduu = -phiMax * phiMax * vec3f(pHit.x, pHit.y, 0);
    vec3f d2Pduv =
        (thetaMax - thetaMin) * pHit.z * phiMax * vec3f(-sinPhi, cosPhi, 0.);
    vec3f d2Pdvv = -(thetaMax - thetaMin) * (thetaMax - thetaMin) *
        vec3f(pHit.x, pHit.y, pHit.z);
    
    Float E = Dot(dpdu, dpdu);
    Float F = Dot(dpdu, dpdv);
    Float G = Dot(dpdv, dpdv);
    vec3f N = Normalize(Cross(dpdu, dpdv));
    Float e = Dot(N, d2Pduu);
    Float f = Dot(N, d2Pduv);
    Float g = Dot(N, d2Pdvv);
    
    Float invEGF2 = 1 / (E * G - F * F);
    Normal3f dndu = Normal3f((f * F - e * G) * invEGF2 * dpdu +
                             (e * F - f * E) * invEGF2 * dpdv);
    Normal3f dndv = Normal3f((g * F - f * G) * invEGF2 * dpdu +
                             (f * F - g * E) * invEGF2 * dpdv);
    
    vec3f pError = gamma(5) * Abs((vec3f)pHit);
    
    *isect = (ObjectToWorld)(SurfaceInteraction(pHit, pError, Point2f(u, v),
                                                -ray.d, dpdu, dpdv, dndu, dndv,
                                                ray.time, this));
    
    *tHit = (Float)tShapeHit;
    return true;
}