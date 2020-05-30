#pragma once
#include <geometry.h>

template <typename Predicate> inline __bidevice__
int FindInterval(int size, const Predicate &pred){
    int first = 0, len = size;
    while(len > 0){
        int half = len >> 1, middle = first + half;
        if(pred(middle)){
            first = middle + 1;
            len -= half + 1;
        }else
            len = half;
    }
    
    return Clamp(first - 1, 0, size - 2);
}

struct Distribution1D{
    Float *func, *cdf;
    Float funcInt;
    int size;
    
    __bidevice__ int Count() const { return size; }
    
    __bidevice__ Float SampleContinuous(Float u, Float *pdf, int *off = nullptr) const{
        int offset = FindInterval(size, [&](int index) { return cdf[index] <= u; });
        if (off) *off = offset;
        Float du = u - cdf[offset];
        if ((cdf[offset + 1] - cdf[offset]) > 0){
            du /= (cdf[offset + 1] - cdf[offset]);
        }
        
        AssertA(!IsNaN(du), "SampleCountinuous");
        
        if (pdf) *pdf = (funcInt > 0) ? func[offset] / funcInt : 0;
        
        return (offset + du) / Count();
    }
    
    __bidevice__ int SampleDiscrete(Float u, Float *pdf = nullptr,
                                    Float *uRemapped = nullptr) const
    {
        int offset = FindInterval(size, [&](int index) { return cdf[index] <= u; });
        if (pdf) *pdf = (funcInt > 0) ? func[offset] / (funcInt * Count()) : 0;
        if (uRemapped)
            *uRemapped = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);
        return offset;
    }
    
    __bidevice__ Float DiscretePDF(int index) const{
        AssertA(index >= 0 && index < Count(), "Invalid index");
        return func[index] / (funcInt * Count());
    }
};

class Distribution2D{
    public:
    Distribution1D **pConditionalV;
    Distribution1D *pMarginal;
    
    __bidevice__ Point2f SampleContinuous(const Point2f &u, Float *pdf) const{
        Float pdfs[2];
        int v;
        Float d1 = pMarginal->SampleContinuous(u[1], &pdfs[1], &v);
        Float d0 = pConditionalV[v]->SampleContinuous(u[0], &pdfs[0]);
        *pdf = pdfs[0] * pdfs[1];
        return Point2f(d0, d1);
    }
    
    __bidevice__ Float Pdf(const Point2f &p) const {
        int iu = Clamp(int(p[0] * pConditionalV[0]->Count()), 0,
                       pConditionalV[0]->Count() - 1);
        int iv = Clamp(int(p[1] * pMarginal->Count()), 0, pMarginal->Count() - 1);
        return pConditionalV[iv]->func[iu] / pMarginal->funcInt;
    }
};

__host__ Distribution2D *CreateDistribution2D(Float *func, int nu, int nv);