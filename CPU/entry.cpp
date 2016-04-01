#include <iostream>
#include <string>
#include <fstream>
#include "material.h"
#include "hitable_list.h"
#include "camera.h"

const int MAX_BOUNCES = 10;

vec3 color( const ray &r, hitable *world, int depth )
{
	hit_record rec;
	if(world->hit(r, 0.001, FLT_MAX, rec))
	{
		ray scattered;
		vec3 attenuation;
		if(depth < MAX_BOUNCES && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
		{
			return attenuation*color(scattered, world, depth + 1);
		}
		else
			return vec3(0.0, 0.0, 0.0);
	}
	vec3 unit_dir = unit_vector(r.direction());
	float t = 0.5*(unit_dir.y() + 1.0);
	return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

hitable * random_scene()
{
	int n = 500;
    hitable **list = new hitable*[n+1];
    list[0] =  new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    for (int a = -1; a < 1; a++) {
        for (int b = -1; b < 1; b++) {
            float choose_mat = drand48();
            vec3 center(a+0.9*drand48(),0.2,b+0.9*drand48()); 
            if ((center-vec3(4,0.2,0)).length() > 0.9) { 
                if (choose_mat < 0.8) {  // diffuse
                    list[i++] = new sphere(center, 0.2, new lambertian(vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48())));
                }
                else if (choose_mat < 0.95) { // metal
                    list[i++] = new sphere(center, 0.2,
                            new metal(vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())),  0.5*drand48()));
                }
                else {  // glass
                    list[i++] = new sphere(center, 0.2, new dieletric(1.5));
                }
            }
        }
    }

    list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dieletric(1.5));
    list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

    return new hitable_list(list,i);
}

int main(int argc, char **argv)
{
	int nx = 800;
	int ny = 600;
	int ns = 20;
	std::ofstream file("imagem.ppm");
	file << "P3\n" << nx << " " << ny << "\n255\n";

	hitable *world = random_scene();
	vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    camera cam(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);
	std::cout << "Running path tracer algorithm" << std::endl;
	clock_t begin = clock();
	for(int j = ny-1; j >= 0; j--)
	{
		for(int i = 0; i < nx; i++)
		{
			vec3 col = vec3(0.0, 0.0, 0.0);
			for(int s = 0; s < ns; s++)
			{
				float u = float(i + drand48())/float(nx);
				float v = float(j + drand48())/float(ny);
				ray r = cam.get_ray(u, v);
				col += color(r, world, 0);
			}
			col /= float(ns);
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
			int ir = int(255.99*col[0]);
			int ig = int(255.99*col[1]);
			int ib = int(255.99*col[2]);
			file << ir << " " << ig << " " << ib << "\n";
		}
	}
	clock_t end = clock();
	file.close();
	double sec = double(end - begin)/CLOCKS_PER_SEC;
	std::cout << "Ended process of path tracing" << std::endl;
	printTime(static_cast<long>(sec));
	getchar();
	return 0;
}