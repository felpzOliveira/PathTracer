#include <shape.h>

__bidevice__ Box::Box(const Transform &toWorld, Float sizex, Float sizey, Float sizez,
                      bool reverseOrientation)
: Shape(toWorld, reverseOrientation), sizex(sizex), sizey(sizey), sizez(sizez)
{
    type = ShapeType::BOX;
    rects = new Rectangle*[6];
    Transform identity;
    
    Float hx = sizex/2;
    Float hy = sizey/2;
    Float hz = sizez/2;
    
    Transform left  = Translate(hx,0,0)  * RotateY(90);
    Transform right = Translate(-hx,0,0) * RotateY(-90);
    Transform top   = Translate(0,hy,0)  * RotateX(90);
    Transform bot   = Translate(0,-hy,0) * RotateX(-90);
    Transform front = Translate(0,0,hz);
    Transform back  = Translate(0,0,-hz);
    rects[0] = new Rectangle(left, sizez, sizey, reverseOrientation);
    rects[1] = new Rectangle(right, sizez, sizey, reverseOrientation);
    rects[2] = new Rectangle(top, sizex, sizez, reverseOrientation);
    rects[3] = new Rectangle(bot, sizex, sizez, reverseOrientation);
    rects[4] = new Rectangle(front, sizex, sizey, reverseOrientation);
    rects[5] = new Rectangle(back, sizex, sizey, reverseOrientation);
}

__bidevice__ bool Box::Intersect(const Ray &r, Float *tHit,
                                 SurfaceInteraction *isect) const
{
    bool hit = false;
    vec3f oErr, dErr;
    SurfaceInteraction tmp;
    Ray ray = WorldToObject(r, &oErr, &dErr);
    Float t;
    for(int i = 0; i < 6; i++){
        if(rects[i]->Intersect(ray, &t, &tmp)){
            ray.tMax = t; hit = true;
        }
    }
    
    if(hit){
        *isect = ObjectToWorld(tmp);
        *tHit = t;
    }
    
    return hit;
}

__bidevice__ Bounds3f Box::GetBounds() const{
    Float hx = sizex/2;
    Float hy = sizey/2;
    Float hz = sizez/2;
    return ObjectToWorld(Bounds3f(Point3f(-hx,-hy,-hz), 
                                  Point3f(hx,hy,hz)));
}

__bidevice__ Float Box::Area() const{
    return 2 * (sizex * sizey + sizex * sizez + sizey * sizez);
}

