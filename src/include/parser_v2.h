#if !defined(PARSER_V2_H)
#define PARSER_V2_H

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <thread>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <vector>
#include <ctime>

#define Timed(op, x) do{\
    std::cout << ">> Running: " << op << std::endl;\
    clock_t start = clock();\
    { x; }\
    clock_t end = clock();\
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;\
    std::string unit("s");\
    if(cpu_time_used > 60){\
        cpu_time_used /= 60.0f;\
        unit = "min";\
    }\
    if(cpu_time_used > 60){\
        cpu_time_used /= 60.0f;\
        unit = "h";\
    }\
    std::cout << "Took >> " << cpu_time_used << " " << unit << std::endl;\
}while(0)

class DelayedThread{
    public:
    std::thread threadId;
    bool active;
    
    DelayedThread(){
        active = false;
    }
    
    template<class Fn, class... Args> bool launch(Fn &&fn, Args&&... args){
        threadId = std::thread{ [&] (){
                active = true;
                fn(args...);
                active = false;
            }
        };
        
        threadId.detach();
        return true;
    }
};

typedef enum{
    Vector=0, Scalar
}Types;

typedef struct{
    std::vector<std::vector<glm::vec3> *> vector_data;
    std::vector<std::vector<float> *> scalar_data;
    int index;
}ParserData;

typedef struct{
    std::vector<ParserData *> frames;
    std::vector<Types> types;
    int activeFrame;
    int hasScalar;
    int hasVector;
}ParsedBlob;

typedef struct{
    ParsedBlob blob;
}Parser_v2;


Parser_v2 * Parser_v2_new(const char *format);
void Parser_v2_load_single_file(Parser_v2 *parser, const char *name);
void Parser_v2_load_multiple_files(Parser_v2 * parser, const char *basename,
                                   const char *ext, int start, int end,
                                   int maxthreads=10);

void Parser_v2_load_multiple_filesv(Parser_v2 *parser, const char *basename,
                                    const char *ext, std::vector<int> steps,
                                    int maxthreads=10);

std::vector<glm::vec3> * Parser_v2_get_vector_ptr(Parser_v2 *parser, int i);
std::vector<float> * Parser_v2_get_scalar_ptr(Parser_v2 *parser, int i);

std::vector<glm::vec3> * Parser_v2_get_vector_ptr(Parser_v2 *parser, int frame, int i);
std::vector<float> * Parser_v2_get_scalar_ptr(Parser_v2 *parser, int frame, int i);

glm::vec3 * Parser_v2_get_raw_vector_ptr(Parser_v2 *parser, int frame, 
                                         int i, size_t *count);
float * Parser_v2_get_raw_scalar_ptr(Parser_v2 *parser, int frame,
                                     int i, size_t *count);

#endif