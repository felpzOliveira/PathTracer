#include <part_loader.h>
#include <obj_loader.h>
#include <mtl.h>
#include <fstream>

#define STATE_INITIALIZED  1
#define STATE_ALLOCATED    2

static __host__ bool IsVector3(char ref){
    return ref == 'v' || ref == 'V';
}

static __host__ bool IsScalar(char ref){
    return ref == 's' || ref == 'S';
}

static __host__ void PParseAllocateFor(PParser *parser, int pCount){
    if(parser->state == STATE_INITIALIZED){
        int vcount = 0;
        int scount = 0;
        int count = 0;
        for(int i = 0; i < parser->format.size(); i++){
            char ref = parser->format[i];
            if(IsVector3(ref)) vcount ++;
            else if(IsScalar(ref)) scount ++;
            else printf("Skipping unknown format: %c\n", ref);
        }
        
        parser->data = cudaAllocateVx(PParsedData, 1);
        parser->data->vector_data = nullptr;
        parser->data->scalar_data = nullptr;
        if(vcount > 0)
            parser->data->vector_data = cudaAllocateVx(vec3f *, vcount);
        if(scount > 0)
            parser->data->scalar_data = cudaAllocateVx(Float *, scount);
        
        parser->data->nvectors = vcount;
        parser->data->nscalars = scount;
        
        count = Max(vcount, scount);
        for(int i = 0; i < count; i++){
            if(i < vcount)
                parser->data->vector_data[i] = cudaAllocateVx(vec3f, pCount);
            if(i < scount)
                parser->data->scalar_data[i] = cudaAllocateVx(Float, pCount);
        }
        
        parser->data->count = pCount;
        parser->state = STATE_ALLOCATED;
    }else{
        printf("Cannot allocate uninitialized parser\n");
    }
}

static __host__ bool PParseFile(PParser *parser, const char *path){
    bool rv = false;
    std::ifstream ifs(path);
    if(!ifs){
        printf("Failed to open file: %s\n", path);
        return rv;
    }
    
    std::string linebuf;
    int firstline = 1;
    int partCount = 0;
    int pid = -1;
    clock_t start = clock();
    
    while(ifs.peek() != -1){
        GetLine(ifs, linebuf);
        
        if(linebuf.size() > 0){ //remove '\n'
            if(linebuf[linebuf.size()-1] == '\n') linebuf.erase(linebuf.size() - 1);
        }
        
        if(linebuf.size() > 0){ //remove '\r'
            if(linebuf[linebuf.size()-1] == '\r') linebuf.erase(linebuf.size() - 1);
        }
        
        // skip empty
        if(linebuf.empty()) continue;
        const char *token = linebuf.c_str();
        token += strspn(token, " \t");
        
        Assert(token);
        
        if(token[0] == '\0') continue; //empty line
        if(token[0] == '#') continue; //comment line
        
        if(firstline){ // first line is number of particles
            partCount = static_cast<int>(ParseFloat(&token));
            printf("Particle count %d\n", partCount);
            firstline = 0;
            if(partCount < 1) return false;
            PParseAllocateFor(parser, partCount);
        }else{
            // get the floats
            char ref = 0;
            int vid = 0;
            int sid = 0;
            pid++;
            // for every option given
            for(int i = 0; i < parser->format.size(); i++){
                ref = parser->format[i];
                if(IsVector3(ref)){
                    vec3f v;
                    vec3f *ptr = parser->data->vector_data[vid++];
                    ParseV3(&v, &token);
                    ptr[pid] = v;
                }else if(IsScalar(ref)){
                    Float s = ParseFloat(&token);
                    Float *ptr = parser->data->scalar_data[sid++];
                    ptr[pid] = s;
                }else{
                    // skip unknown arg
                }
            }
        }
    }
    
    pid++;
    rv = (partCount == pid);
    
    clock_t end = clock();
    
    double time_taken = to_cpu_time(start, end);
    printf("Took %g seconds, #p [%d] #v [%d] #s [%d].\n", 
           time_taken, pid, parser->data->nvectors, parser->data->nscalars);
    return rv;
}

__host__ void PParserCleanup(PParser *parser){
    if(parser->state == STATE_ALLOCATED){
        int it = Max(parser->data->nvectors, parser->data->nscalars);
        for(int i = it-1; i >= 0; i--){
            if(i < parser->data->nvectors)
                cudaFree(parser->data->vector_data[i]);
            if(i < parser->data->nscalars)
                cudaFree(parser->data->scalar_data[i]);
        }
        
        if(parser->data->vector_data) cudaFree(parser->data->vector_data);
        if(parser->data->scalar_data) cudaFree(parser->data->scalar_data);
        
        cudaFree(parser->data);
        parser->data = nullptr;
        parser->state = STATE_INITIALIZED;
    }
}

__host__ void PParserInitialize(PParser *parser, const char *format){
    parser->format = std::string(format);
    parser->data = nullptr;
    parser->state = STATE_INITIALIZED;
}

__host__ bool PParserParse(PParser *parser, const char *filename){
    bool rv = false;
    if(parser->state == STATE_INITIALIZED && parser->data == nullptr){
        rv = PParseFile(parser, filename);
    }else{
        printf("Invalid parser setup\n");
    }
    
    return rv;
}

__host__ vec3f *PParserGetVectorPtr(PParser *parser, int index, int *count){
    vec3f *ptr = nullptr;
    *count = 0;
    if(parser->state == STATE_ALLOCATED){
        if(parser->data->nvectors > index){
            ptr = parser->data->vector_data[index];
            *count = parser->data->count;
        }
    }
    
    return ptr;
}

__host__ Float *PParserGetScalarPtr(PParser *parser, int index, int *count){
    Float *ptr = nullptr;
    *count = 0;
    if(parser->state == STATE_ALLOCATED){
        if(parser->data->nscalars > index){
            ptr = parser->data->scalar_data[index];
            *count = parser->data->count;
        }
    }
    
    return ptr;
}