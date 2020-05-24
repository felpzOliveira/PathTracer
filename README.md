# Toy-Tracer
This is my personal Path Tracer code. I'm writing this as I read PBRT, trying my best to adapt to run with CUDA for speedup and learning purposes.
It is actually going really good, have almost the full Path Tracer with light sampling working, I'm still missing the medium
interface which I hope to implement soon, PBRT has some really good-looking images based on mediums so this is a must.

This is also based on previous version which I did based on the mini-series by Peter Shirley and the Advanced Global Illumination book,
so I remade a few images from those using the new code, but now we have a bunch of new stuff with the BSDF interface and the Microfacet distribution which can generate really good images:

![Alt text](/images/room.png)
![Alt text](/images/scene0.png)
![Alt text](/images/cornell.png) 


As I develop, this became my personal favorite thing to render with. I also work with Fluids (SPH, PIC, APIC, Boundaries, etc...)
and this is the code I use to render some fluid images:

![Alt text](/images/quad_dam_80.png)
![Alt text](/images/fluid_scene1.png)

The lovely dragon from PBRT, this one actually has a bug from when I was developing it, I only caught it later on (and so the noisy image) but I find
it lovely still, so let it be :)

![Alt text](/images/dragon2.png)
![Alt text](/images/glassmicro.png)
