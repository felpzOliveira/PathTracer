#include "PathTracer.cuh"

#define RESOLUTION_X 1024
#define RESOLUTION_Y 860
#define SAMPLES_PER_PIXEL 20
#define SAMPLES_PER_BATCH 10

const float ASPECT_RATIO = float(RESOLUTION_X)/float(RESOLUTION_Y);

Scene * cornell_scene( void )
{
	vec3 cam_eye(0, 2, 0);
	vec3 cam_at(0, 2, -5);
	vec3 cam_up(0,1,0);

	Scene *sc = new Scene(new Camera(cam_eye, cam_at, cam_up, ASPECT_RATIO), 1, 0, 0, 2);
	sc->add(new Plane(vec3(0,1,0), vec3(0,1,0), new Material(new Texture(CONSTANT,
															 vec3(0.78, 0.78, 0.78)), DIFFUSE)));
	sc->add(new Plane(vec3(0,0, 5), vec3(0,0,-1), new Material(new Texture(CONSTANT,
															 vec3(0.78, 0.78, 0.12)), DIFFUSE)));
	sc->add(new Sphere(vec3(0,3,0), 1, new Material(new Texture(CONSTANT, vec3(1,1,1)), LIGHT)));

	return sc;
}

int main(int argc, char **argv)
{
	std::cout << "Loading scene...";
	Scene *sc = cornell_scene();
	std::cout << "finished." << std::endl;
	PathTracer *tracer = new PathTracer(RESOLUTION_X, RESOLUTION_Y, SAMPLES_PER_PIXEL,
										SAMPLES_PER_BATCH, INTEROP::PTRACER_NO_GL_INTEROP);
	tracer->Render(sc);
	tracer->DisplayRenderWithGimp();
}