#pragma once

#include <shape.h>

class Primitive{
    public:
    __bidevice__ virtual bool Intersect(const Ray &r, SurfaceInteraction *) const = 0;
    __bidevice__ virtual void Release() const = 0;
};

class GeometricPrimitive : public Primitive{
    public:
    Shape *shape;
    __bidevice__ GeometricPrimitive(){}
    __bidevice__ GeometricPrimitive(Shape *shape);
    __bidevice__ virtual bool Intersect(const Ray &r, SurfaceInteraction *) const override;
    __bidevice__ virtual void Release() const override{
        delete shape;
    }
};

class AggregateList{
    public:
    Primitive **primitives;
    int length;
    int head;
    __bidevice__ AggregateList(int size);
    __bidevice__ void Insert(Primitive *pri);
    __bidevice__ bool Intersect(const Ray &r, SurfaceInteraction *) const;
    __bidevice__ void Release();
};