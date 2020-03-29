#if !defined(FLUID_CUT_H)
#define FLUID_CUT_H
#include <glm/glm.hpp>

//#define FLUID_SCENE_BALL
#define FLUID_SCENE_DDB

#ifdef FLUID_SCENE_BALL
/* BALL scene */
static glm::vec3 fOrigin = glm::vec3(0.8f, 2.0f, 3.7f);
static glm::vec3 fTarget = glm::vec3(1.0f);
static glm::vec3 fCutSource(1.5f, 1.0f, 3.7f);
static float fCutDistance = 2.1f;
static float fPlaneHeight = -0.5f;
////////////////////////////////////////////////////////////
#elif defined(FLUID_SCENE_DDB)
static glm::vec3 fOrigin = glm::vec3(1.5f, 1.8f,-2.5f);
static glm::vec3 fTarget = glm::vec3(1.5f,1.0f,0.75f);
static glm::vec3 fCutSource(1.5f, 1.8f,-3.7f);
static float fCutDistance = 4.4f;
static float fPlaneHeight = 0.0f;
////////////////////////////////////////////////////////////
#else
static glm::vec3 fOrigin = glm::vec3(0.8f, 2.0f, 3.7f);
static glm::vec3 fTarget = glm::vec3(1.0f);
static glm::vec3 fCutSource(1.5f, 1.0f, 3.7f);
static float fCutDistance = 2.1f;
static float fPlaneHeight = -0.5f;
#endif

int cut_fluid_particle(glm::vec3 pi){
    float d = glm::distance(pi, fCutSource);
    return (d > fCutDistance ? 0 : 1);
}

#endif