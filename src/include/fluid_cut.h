#if !defined(FLUID_CUT_H)
#define FLUID_CUT_H
#include <glm/glm.hpp>

#define FLUID_SCENE_QUAD_DAM
//#define FLUID_SCENE_BALL
//#define FLUID_SCENE_DDB
//#define FLUID_SCENE_WATER_DROP

#ifdef FLUID_SCENE_BALL
/* BALL scene */
//#define SRC_PATH "/home/felpz/Documents/Fluids/boundaries/gpu/output/"
#define SRC_PATH "/home/felpz/SimplexSphere2/"
static const char *fPath = SRC_PATH "OUT_PART_SimplexSphere2_120.txt";
static glm::vec3 fOrigin = glm::vec3(0.8f, 2.0f, 3.7f);
static glm::vec3 fTarget = glm::vec3(1.0f);
static glm::vec3 fCutSource(1.5f, 1.0f, 3.7f);
static float fCutDistance = 2.55f;
static float fPlaneHeight = -0.5f;
////////////////////////////////////////////////////////////
#elif defined(FLUID_SCENE_DDB)
#define SRC_PATH "/home/felpz/"
static const char *fPath = SRC_PATH "out2_0.txt";
//static const char *fPath = SRC_PATH "OUT_PART_3DRun_50.txt";
//static glm::vec3 fOrigin = glm::vec3(1.5f, 1.8f,-2.5f);
static glm::vec3 fOrigin = glm::vec3(1.5f, 1.8f,-4.2f);
static glm::vec3 fTarget = glm::vec3(1.5f,1.3f,0.75f);
static glm::vec3 fCutSource(1.5f, 1.8f,-3.7f);
static float fCutDistance = -4.4f;
static float fPlaneHeight = 0.0f;
////////////////////////////////////////////////////////////
#elif defined(FLUID_SCENE_QUAD_DAM)
#define SRC_PATH "/home/felpz/Documents/Fluids/boundaries/gpu/output/"
//#define SRC_PATH "/home/felpz/"
static const char *fPath = SRC_PATH "out2_0.txt";
static glm::vec3 fOrigin = glm::vec3(1.5f, 1.8f,-3.85f);
static glm::vec3 fTarget = glm::vec3(1.5f,1.5f,0.75f);
static glm::vec3 fCutSource(1.5f, 1.8f,-3.7f);
static float fCutDistance = 5.21f;
static float fPlaneHeight = 0.0f;
////////////////////////////////////////////////////////////
#elif defined(FLUID_SCENE_WATER_DROP)
#define SRC_PATH "/home/felpz/"
static const char *fPath = SRC_PATH "OUT_PART_ClassesWD_26.txt";
static glm::vec3 fOrigin = glm::vec3(1.5f, 1.8f,-4.85f);
static glm::vec3 fTarget = glm::vec3(1.5f,1.5f,0.75f);
static glm::vec3 fCutSource(1.5f, 1.8f,-3.7f);
static float fCutDistance = 4.51f;
static float fPlaneHeight = 0.0f;
////////////////////////////////////////////////////////////
#else
static const char *fPath = "fail.txt";
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
