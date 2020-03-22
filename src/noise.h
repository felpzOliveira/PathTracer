#if !defined(NOISE_H)
#define NOISE_H

#include <types.h>

inline __host__ __device__ glm::vec4 permute(glm::vec4 x){
    float sx = glm::mod(((x.x*34.0)+1.0)*x.x, 289.0);
    float sy = glm::mod(((x.y*34.0)+1.0)*x.y, 289.0);
    float sz = glm::mod(((x.z*34.0)+1.0)*x.z, 289.0);
    float sw = glm::mod(((x.w*34.0)+1.0)*x.w, 289.0);
    return glm::vec4(sx, sy, sz, sw);
}

inline __host__ __device__ 
glm::vec4 taylorInvSqrt(glm::vec4 r){return 1.79284291400159f-0.85373472095314f * r;}

inline __host__ __device__ float snoise(glm::vec3 v){ 
    const glm::vec2  C = glm::vec2(1.0/6.0, 1.0/3.0) ;
    const glm::vec4  D = glm::vec4(0.0, 0.5, 1.0, 2.0);
    
    // First corner
    glm::vec3 i  = glm::floor(v + glm::dot(v, glm::vec3(C.y)) );
    glm::vec3 x0 =   v - i + glm::dot(i, glm::vec3(C.x)) ;
    
    // Other corners
    glm::vec3 g = glm::step(glm::vec3(x0.y, x0.z, x0.x), 
                            glm::vec3(x0.x, x0.y, x0.z));
    glm::vec3 l = 1.0f - g;
    glm::vec3 i1 = glm::min( glm::vec3(g.x,g.y,g.z), glm::vec3(l.z,l.x,l.y) );
    glm::vec3 i2 = glm::max( glm::vec3(g.x,g.y,g.z), glm::vec3(l.z,l.x,l.y) );
    
    //  x0 = x0 - 0. + 0.0 * C 
    glm::vec3 x1 = x0 - i1 + 1.0f * glm::vec3(C.x);
    glm::vec3 x2 = x0 - i2 + 2.0f * glm::vec3(C.x);
    glm::vec3 x3 = x0 - 1.f + 3.0f * glm::vec3(C.x);
    
    // Permutations
    i = glm::mod(i, 289.0f ); 
    glm::vec4 p = permute( permute( permute( 
        i.z + glm::vec4(0.0, i1.z, i2.z, 1.0 ))
                                   + i.y + glm::vec4(0.0, i1.y, i2.y, 1.0 )) 
                          + i.x + glm::vec4(0.0, i1.x, i2.x, 1.0 ));
    
    // Gradients
    // ( N*N points uniformly over a square, mapped onto an octahedron.)
    float n_ = 1.0/7.0; // N=7
    glm::vec3  ns = n_ * glm::vec3(D.w, D.y,D.z) - glm::vec3(D.x,D.z,D.x);
    
    glm::vec4 j = p - 49.0f * glm::floor(p * ns.z *ns.z);  //  mod(p,N*N)
    
    glm::vec4 x_ = glm::floor(j * ns.z);
    glm::vec4 y_ = glm::floor(j - 7.0f * x_ );    // mod(j,N)
    
    glm::vec4 x = x_ *ns.x + glm::vec4(ns.y);
    glm::vec4 y = y_ *ns.x + glm::vec4(ns.y);
    glm::vec4 h = 1.0f - glm::abs(x) - glm::abs(y);
    
    glm::vec4 b0 = glm::vec4( x.x,x.y, y.x,y.y );
    glm::vec4 b1 = glm::vec4( x.z,x.w, y.z,y.w );
    
    glm::vec4 s0 = glm::floor(b0)*2.0f + 1.0f;
    glm::vec4 s1 = glm::floor(b1)*2.0f + 1.0f;
    glm::vec4 sh = -glm::step(h, glm::vec4(0.0f));
    
    glm::vec4 a0 = glm::vec4(b0.x,b0.z,b0.y,b0.w) + glm::vec4(s0.x,s0.z,s0.y,s0.w)*glm::vec4(sh.x,sh.x,sh.y,sh.y) ;
    glm::vec4 a1 = glm::vec4(b1.x,b1.z,b1.y,b1.w) + glm::vec4(s1.x,s1.z,s1.y,s1.w)*glm::vec4(sh.z,sh.z,sh.w,sh.w) ;
    
    glm::vec3 p0 = glm::vec3(a0.x,a0.y,h.x);
    glm::vec3 p1 = glm::vec3(a0.z,a0.w,h.y);
    glm::vec3 p2 = glm::vec3(a1.x,a1.y,h.z);
    glm::vec3 p3 = glm::vec3(a1.z,a1.w,h.w);
    
    //Normalise gradients
    glm::vec4 norm = taylorInvSqrt(glm::vec4(glm::dot(p0,p0), 
                                             glm::dot(p1,p1), 
                                             glm::dot(p2, p2), 
                                             glm::dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    
    // Mix final noise value
    float v0 = glm::max(0.6f - glm::dot(x0,x0), 0.0f);
    float v1 = glm::max(0.6f - glm::dot(x1,x1), 0.0f);
    float v2 = glm::max(0.6f - glm::dot(x2,x2), 0.0f);
    float v3 = glm::max(0.6f - glm::dot(x3,x3), 0.0f);
    glm::vec4 m(v0, v1, v2, v3);
    m = m * m;
    return 42.0 * glm::dot( m*m, glm::vec4( glm::dot(p0,x0), glm::dot(p1,x1), 
                                           glm::dot(p2,x2), glm::dot(p3,x3) ) );
}

inline __host__ __device__ float trilinear_interp(glm::vec3 c[2][2][2], 
                                                  float u, float v, float w)
{
    float uu = u*u*(3-2*u);
    float vv = v*v*(3-2*v);
    float ww = w*w*(3-2*w);
    float accum = 0;
    for (int i=0; i < 2; i++)
        for (int j=0; j < 2; j++)
        for (int k=0; k < 2; k++) {
        glm::vec3 weight_v(u-i, v-j, w-k);
        accum += (i*uu + (1-i)*(1-uu))*
            (j*vv + (1-j)*(1-vv))*
            (k*ww + (1-k)*(1-ww))*glm::dot(c[i][j][k], weight_v);
    }
    return accum;
}



inline __host__ __device__ float noise31(Perlin *perlin, glm::vec3 p, int trilinear){
    float u = p.x - glm::floor(p.x);
    float v = p.y - glm::floor(p.y);
    float w = p.z - glm::floor(p.z);
    
    int max = perlin->size - 1;
    if(trilinear){
        float u = p.x - glm::floor(p.x);
        float v = p.y - glm::floor(p.y);
        float w = p.z - glm::floor(p.z);
        int i = glm::floor(p.x);
        int j = glm::floor(p.y);
        int k = glm::floor(p.z);
        glm::vec3 c[2][2][2];
        for (int di=0; di < 2; di++)
            for (int dj=0; dj < 2; dj++)
            for (int dk=0; dk < 2; dk++)
            c[di][dj][dk] = perlin->ranvec[perlin->permx[(i+di) & 255] ^ 
                perlin->permy[(j+dj) & 255] ^
                perlin->permz[(k+dk) & 255]];
        return trilinear_interp(c, u, v, w);
        
    }else{
        int i = int(4.0f*p.x) & max;
        int j = int(4.0f*p.y) & max;
        int k = int(4.0f*p.z) & max;
        glm::vec3 weight(u-i, v-j, w-k);
        int rng = perlin->permx[i] ^ perlin->permy[j] ^ perlin->permz[k];
        return glm::dot(perlin->ranvec[rng], weight);
    }
}

inline __host__ __device__ float turb(Perlin *perlin, glm::vec3 p, int depth=7){
    float accum = 0.0f;
    glm::vec3 tmp_p = p;
    float weight = 1.0f;
    for(int i = 0; i < depth; i += 1){
        accum += weight * noise31(perlin, tmp_p, 1);
        weight *= 0.5f;
        tmp_p *= 2;
    }
    
    return glm::abs(accum);
}

#endif