#if !defined(TEXTURE_H)
#define TEXTURE_H

#include <types.h>
#include <noise.h>
#include <cutil.h>

inline __bidevice__
Spectrum texture_value(Texture *texture, hit_record *record, Scene *scene);

#include <detail/texture-inl.h>

#endif