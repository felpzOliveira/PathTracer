#include <primitive.h>

__bidevice__ GeometricPrimitive::GeometricPrimitive(Shape *shape)
: shape(shape) {}

__bidevice__ bool GeometricPrimitive::Intersect(const Ray &ray, 
                                                SurfaceInteraction *isect) const
{
    Float tHit;
    if(!shape->Intersect(ray, &tHit, isect)) return false;
    ray.tMax = tHit;
    isect->primitive = this;
    return true;
}


__bidevice__ AggregateList::AggregateList(int size){
    Assert(size > 0);
    length = size;
    head = 0;
    primitives = new Primitive*[size];
}

__bidevice__ void AggregateList::Insert(Primitive *pri){
    Assert(head < length && primitives);
    primitives[head++] = pri;
}

__bidevice__ bool AggregateList::Intersect(const Ray &r, SurfaceInteraction *isect) const{
    bool hit_anything = false;
    Assert(head > 0);
    for(int i = 0; i < head; i++){
        Primitive *pri = primitives[i];
        hit_anything |= pri->Intersect(r, isect);
    }
    
    return hit_anything;
}

__bidevice__ void AggregateList::Release(){
    for(int i = 0; i < head; i++){
        Primitive *pri = primitives[i];
        pri->Release();
    }
    
    delete[] primitives;
}