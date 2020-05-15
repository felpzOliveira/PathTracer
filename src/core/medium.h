#pragma once

#include <geometry.h>
#include <reflection.h>

//TODO
class PhaseFunction{
    public:
    const Float g;
    __bidevice__ PhaseFunction(Float g) : g(g){}
    
};