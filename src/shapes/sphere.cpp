#include <shape.h>

__bidevice__ Bounds3f Sphere::GetBounds() const{
    return (ObjectToWorld)(Bounds3f(Point3f(-radius, -radius, zMin),
                                    Point3f(radius, radius, zMax)));
}

__bidevice__ Float Sphere::Area() const{
    return phiMax * radius * (zMax - zMin);
}

__bidevice__ Interaction Sphere::Sample(const Point2f &u, Float *pdf) const{
    Point3f pObj = Point3f(0) + radius * SampleSphere(u);
    Interaction it;
    it.n = Normalize(ObjectToWorld(Normal3f(pObj.x, pObj.y, pObj.z)));
    //if(reverseOrientation) it.n *= -1;
    
    pObj *= radius / Distance(pObj, Point3f(0, 0, 0));
    vec3f pObjError = gamma(5) * Abs(ToVec3(pObj));
    it.p = ObjectToWorld(pObj, pObjError, &it.pError);
    *pdf = 1 / Area();
    return it;
}

__bidevice__ Interaction Sphere::Sample(const Interaction &ref, const Point2f &u,
                                        Float *pdf) const
{
    Point3f pCenter = ObjectToWorld(Point3f(0, 0, 0));
    Point3f pOrigin = OffsetRayOrigin(ref.p, ref.pError, ref.n, pCenter - ref.p);
    if(DistanceSquared(pOrigin, pCenter) <= radius * radius){
        Interaction intr = Sample(u, pdf);
        vec3f wi = intr.p - ref.p;
        if(IsZero(wi.LengthSquared())){
            *pdf = 0;
        }else{
            wi = Normalize(wi);
            *pdf *= DistanceSquared(ref.p, intr.p) / AbsDot(intr.n, -wi);
        }
        if (std::isinf(*pdf)) *pdf = 0.f;
        return intr;
    }
    
    Float dc = Distance(ref.p, pCenter);
    Float invDc = 1 / dc;
    vec3f wc = (pCenter - ref.p) * invDc;
    vec3f wcX, wcY;
    CoordinateSystem(wc, &wcX, &wcY);
    
    Float sinThetaMax = radius * invDc;
    Float sinThetaMax2 = sinThetaMax * sinThetaMax;
    Float invSinThetaMax = 1 / sinThetaMax;
    Float cosThetaMax = std::sqrt(Max((Float)0.f, 1 - sinThetaMax2));
    Float cosTheta  = (cosThetaMax - 1) * u[0] + 1;
    Float sinTheta2 = 1 - cosTheta * cosTheta;
    
    if(sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */){
        /* Fall back to a Taylor series expansion for small angles, where
           the standard approach suffers from severe cancellation errors */
        sinTheta2 = sinThetaMax2 * u[0];
        cosTheta = std::sqrt(1 - sinTheta2);
    }
    
    Float cosAlpha = sinTheta2 * invSinThetaMax +
        cosTheta * std::sqrt(Max((Float)0.f, 1.f - sinTheta2 * 
                                 invSinThetaMax * invSinThetaMax));
    
    Float sinAlpha = std::sqrt(Max((Float)0.f, 1.f - cosAlpha*cosAlpha));
    Float phi = u[1] * 2 * Pi;
    vec3f nWorld = SphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
    Point3f pWorld = pCenter + radius * Point3f(nWorld.x, nWorld.y, nWorld.z);
    Interaction it;
    it.p = pWorld;
    it.pError = gamma(5) * Abs((vec3f)pWorld);
    it.n = Normal3f(nWorld);
    //if (reverseOrientation) it.n *= -1;
    *pdf = 1 / (2 * Pi * (1 - cosThetaMax));
    return it;
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