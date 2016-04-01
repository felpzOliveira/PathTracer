#pragma once
#include "cu_math.cuh"
#include "material.cuh"

class Box : public Managed{
private:
	Material *m_mat;
	vec3 m_min;
	vec3 m_max;
public:
	__host__ __device__ Box( vec3 min, vec3 max, Material *mat )
	{ m_mat = mat; m_min = min; m_max = max; }
	__host__ __device__ bool hit( Ray r, float t_min, float t_max, hit_record &rec )
	{
		
		float ox = r.origin().x(), oy = r.origin().y(), oz = r.origin().z();
		float dx = r.direction().x(), dy = r.direction().y(), dz = r.direction().z();
		float x0 = m_min.x(), y0 = m_min.y(), z0 = m_min.z();
		float x1 = m_max.x(), y1 = m_max.y(), z1 = m_max.z();
		
		float tx_min, ty_min, tz_min;
		float tx_max, ty_max, tz_max;
		float a = 1/dx, b = 1/dy, c = 1/dz;
		
		if(a >= 0){ tx_min = (x0 - ox)*a; tx_max = (x1-ox)*a; }
		else { tx_min = (x1-ox)*a; tx_max = (x0-ox)*a; }
		
		if(b >= 0){ ty_min = (y0-oy)*b; ty_max = (y1-oy)*b; }
		else{ ty_min = (y1-oy)*b; ty_max = (y0-oy)*b; }

		if(c >= 0){ tz_min = (z0-oz)*c; tz_max = (z1-oz)*c; }
		else{ tz_min = (z1-oz)*c; tz_max = (z0-oz)*c; }

		float t0,t1; int face_in, face_out;
		if(tx_min > ty_min){ t0 = tx_min; face_in = (a>=0)?0:3; }
		else{ t0 = ty_min; face_in = (b>=0)?1:4; }

		if(tz_min > t0){ t0 = tz_min; face_in = (c>=0)?2:5; }
		
		if(tx_max < ty_max){ t1 = tx_max; face_out = (a>=0)?3:0; }
		else{ t1 = ty_max; face_out = (b>=0)?4:1; }

		if(tz_max < t1){ t1 = tz_max; face_out = (c>=0)?5:2; }
		
		float tmin;
		if(t0 < t1 && t1 > 0){
			if(t0 > 0){
				tmin = t0;
				rec.normal = get_normal(face_in);
			}
			else{
				tmin = t1;
				rec.normal = get_normal(face_out);
			}
			rec.point = r.origin() + tmin*r.direction();
			rec.t = tmin;
			if(t1 > t0 && t1 > 0.0001){
				rec.t = t1;
				if(rec.t > t_min && rec.t < t_max){ rec.mat = m_mat; return true; } 
				
				return false;
			}
		}
		return false;
	}

private:
	__host__ __device__ vec3 get_normal( int face )
	{
		vec3 normal;
		switch(face){
		case 0:normal = vec3(-1,0,0);
		case 1:normal = vec3(0,-1,0);
		case 2:normal = vec3(0,0,-1);
		case 3:normal = vec3(1,0,0);
		case 4:normal = vec3(0,1,0);
		case 5:normal = vec3(0,0,1);
		}
		return normal;
	}
};

class Plane : public Managed{
private:
	Material *m_mat;
	vec3 point;
	vec3 normal;
public:
	__host__ __device__ Plane( vec3 p, vec3 n, Material *mat )
	{ point = p; normal = n; m_mat = mat; }

	__host__ __device__ bool hit( Ray r, float t_min, float t_max, hit_record &rec )
	{
		float denom = dot(normal, r.direction());
		//if(denom > 0.0001f)
		//{
			float t = dot(point - r.origin(), normal)/denom;
			if(t > 0 && t > t_min && t < t_max)
			{
				rec.t = t;
				rec.normal = normal;
				rec.point = r.point_at(rec.t);
				rec.mat = m_mat;
				return true;
			}
		//}
		return false;
	}
};

class Rect : public Managed{
private:
	Material *m_mat;
	vec3 m_p0;
	vec3 m_normal;
	vec3 m_a, m_b;
public:
	__host__ __device__ Rect ( vec3 p0, vec3 a, vec3 b, vec3 norm, Material *mat )
	{m_mat = mat; m_p0 = p0; m_a = a; m_b = b; m_normal = norm; }
	__host__ __device__ bool hit( Ray r, float t_min, float t_max, hit_record &rec )
	{
		float s = dot(r.direction(), m_normal);
		if(s == 0) return false;
		float t = dot(m_p0 - r.origin(), m_normal);
		t = t / s;
		if(t < 0.0001) return false;

		vec3 p = r.origin() + t*r.direction();
		vec3 d = m_p0 - p;
		float ddota = dot(d,m_a);
		if(ddota < 0.0 || ddota > m_a.squared_length())
			return false;

		float ddotb = dot(d, m_b);
		if(ddotb < 0.0 || ddotb > m_b.squared_length())
			return false;

		if(t < t_min || t > t_max) return false;

		rec.t = t;
		if(dot(r.direction(), m_normal) > 0) rec.normal = -m_normal;
		else rec.normal = m_normal;
		rec.mat = m_mat;
		rec.point = p;
		return true;
	}
};

class Sphere : public Managed{
private:
	Material *m_mat;
	float m_radius;
	vec3 m_center;
public:
	__host__ __device__ Sphere( vec3 c, float r, Material *mat ){ m_radius = r; m_center = c; m_mat = mat; }
	__host__ __device__ bool hit( Ray r, float t_min, float t_max, hit_record& rec )
	{
		vec3 or = r.origin() - m_center;
		vec3 dir = r.direction();
		float b = dot(or, dir);
		float c = dot(or, or) - m_radius*m_radius;

		float delt = b*b - c;
		if(delt < 0) return false;

		float t = (-b -sqrt(delt));
		if(t < t_max && t > t_min)
		{
			rec.t = t;
			rec.point = r.point_at(rec.t);
			rec.normal = (rec.point - m_center)/m_radius;
			rec.mat = m_mat;
			return true;
		}
		
		t = (-b + sqrt(delt));
		if(t < t_max && t > t_min)
		{
			rec.t = t;
			rec.point = r.point_at(rec.t);
			rec.normal = (rec.point - m_center)/m_radius;
			rec.mat = m_mat;
			return true;
		}
		return false;

	}
};

class Scene : public Managed{
private:	
	Camera *m_camera;
	Sphere **m_spheres;
	Rect **m_rectangles;
	Box **m_boxes;
	Plane **m_planes;
	int m_rect_i;
	int m_sph_i;
	int m_box_i;
	int m_plane_i;
public:

	__host__ Scene( Camera *camera, int max_sphere_amount, int max_rect_amount, int max_box_amount,
					int max_plane_amount )
	{
		set_camera(camera);
		init_and_cudamalloc(max_sphere_amount, max_rect_amount, max_box_amount, max_plane_amount);
	}

	__host__ Scene( int max_sphere_amount, int max_rect_amount, int max_box_amount, int max_plane_amount ) 
	{ 
		init_and_cudamalloc(max_sphere_amount, max_rect_amount, max_box_amount, max_plane_amount);
	}

	__host__ __device__ Camera *get_camera( void ) { return m_camera; }
	
	__host__ void set_camera( Camera *camera ) { m_camera = camera; }

	__host__ __device__ void add( Sphere *sph )
	{ m_spheres[m_sph_i] = sph; m_sph_i++; }

	__host__ __device__ void add( Plane *pl )
	{ m_planes[m_plane_i] = pl; m_plane_i++; }

	__host__ __device__ void add( Rect *rect )
	{ m_rectangles[m_rect_i] = rect; m_rect_i++; }
	
	__host__ __device__ void add( Box *box )
	{ m_boxes[m_box_i] = box; m_box_i++; }

	__host__ __device__ bool hit( Ray r, float t_min, float t_max, hit_record &rec )
	{
		hit_record temp;
		float max_dist = t_max;
		bool hit_anything = false;

		for(int i = 0; i < m_plane_i; i++){
			if(m_planes[i]->hit(r, t_min, max_dist, temp)){
				max_dist = temp.t;
				rec = temp;
				hit_anything = true;
			}
		}

		for(int i = 0; i < m_sph_i; i++){
			if(m_spheres[i]->hit(r, t_min, max_dist, temp)){
				max_dist = temp.t;
				rec = temp;
				hit_anything = true;
			}
		}

		for(int i = 0; i < m_rect_i; i++){
			if(m_rectangles[i]->hit(r, t_min, max_dist, temp)){
				max_dist = temp.t;
				rec = temp;
				hit_anything = true;
			}
		}
		
		for(int i = 0; i < m_box_i; i++){
			if(m_boxes[i]->hit(r, t_min, max_dist, temp)){
				max_dist = temp.t;
				rec = temp;
				hit_anything = true;
			}
		}
		return hit_anything;
	}
private:
	__host__ void init_and_cudamalloc(int max_sphere_amount, int max_rect_amount, int max_box_amount,
									  int max_plane_amount)
	{
		CHECK(cudaMallocManaged(&m_spheres, sizeof(Sphere*)*max_sphere_amount));
		CHECK(cudaMallocManaged(&m_rectangles, sizeof(Rect*)*max_rect_amount));
		CHECK(cudaMallocManaged(&m_boxes, sizeof(Box*)*max_box_amount));
		CHECK(cudaMallocManaged(&m_planes, sizeof(Plane*)*max_plane_amount));
		m_sph_i = 0; m_rect_i = 0; m_box_i = 0; m_plane_i = 0;
		
	}
};