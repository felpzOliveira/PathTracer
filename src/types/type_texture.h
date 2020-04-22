#if !defined(TYPE_TEXTURE_H)
#define TYPE_TEXTURE_H

#include <glm/glm.hpp>

typedef unsigned int texture_handle;

/* Texture Definition */
typedef enum{
    TEXTURE_CONST,
    TEXTURE_CHECKER,
    TEXTURE_NOISE,
    TEXTURE_IMAGE,
}TextureType;

typedef enum{
    TEXTURE_WRAP_CLAMP,
    TEXTURE_WRAP_REPEAT,
}TextureWrapMode;

typedef enum{
    NOISE_SIMPLE,
    NOISE_TRILINEAR,
}NoiseType;

typedef struct TextureProps_t{
    TextureWrapMode wrap_mode; //wrap mode (clamp to border, repeat ...)
    float scale; //scale for repeating
}TextureProps;

typedef struct Texture_t{
    glm::vec3 color; //for constant texture
    texture_handle odd, even; //checker texture can sample other textures
    NoiseType noise_type; //noise texture have multiple options
    unsigned char *image; //pixels in case this is a image
    int image_x, image_y; //dimensions of the image
    TextureType type; //type of texture (image, color, ...)
    TextureProps props; //properties
}Texture;

#endif