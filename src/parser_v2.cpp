#include "parser_v2.h"
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <chrono>
#define Sleep(x) std::this_thread::sleep_for(std::chrono::seconds(x));
#define Assert(x) if(!(x)){ *(int *)0 = 0; }

template<typename T> static std::string toString(T var){
    std::stringstream ss;
    ss << var;
    return ss.str();
}

static float parser_get_next_float(const char *document, int &mark, int &limiter){
    limiter = -1;
    try {
        int loc = 0;
        char buffer[40];
        int at = mark;
        char chr = document[at];
        while ((chr == ' ' || chr == '\n' || chr == '\r') && chr != '\0') {
            at += 1;
            chr = document[at];
        }
        
        while (chr != ' ' && chr != '\n' && chr != '\r' && chr != '\0') {
            buffer[loc++] = chr;
            at += 1;
            chr = document[at];
        }
        
        buffer[loc] = 0;
        
        if (chr == ' ') {
            limiter = 1;
        }
        else if (chr == '\n' || chr == '\r') {
            limiter = 2;
        }
        else if (chr == '\0') {
            limiter = 3;
        }
        else {
            throw 99;
        }
        mark = at;
        return static_cast<float>(atof(buffer));
    }
    catch (std::exception e) {
        limiter = -1;
        return 0;
    }
}

static void LoadData(const char *path, ParserData *data, std::vector<Types> order){
    Assert(data && path);
    Assert(data->scalar_data.size() + data->vector_data.size() == order.size());
    
    std::ifstream ifs(path);
    if(ifs.is_open()){
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string content = ss.str();
        const char *f = content.c_str();
        
        int c_pointer = 0;
        int v_pointer = 0;
        char curr_char = f[c_pointer++];
        char number[10];
        // get number of particles, first line
        while(curr_char != '\n'){
            number[v_pointer++] = curr_char;
            curr_char = f[c_pointer++];
        }
        number[v_pointer++] = '\0';
        int count = static_cast<int>(atof(number));
        Assert(count > 0);
        
        for(int i = 0; i < data->scalar_data.size(); i += 1){
            data->scalar_data[i]->clear();
        }
        
        for(int i = 0; i < data->vector_data.size(); i += 1){
            data->vector_data[i]->clear();
        }
        
        int processed = 0;
        int position = c_pointer;
        int finished = 0;
        
        while(processed < count && finished != 3){
            int v_it = 0;
            int f_it = 0;
            // parse a single line based on templates
            for(Types &t : order){
                float p1 = parser_get_next_float(f, position, finished);
                if(t == Vector){
                    float p2 = parser_get_next_float(f, position, finished);
                    float p3 = parser_get_next_float(f, position, finished);
                    data->vector_data[v_it++]->push_back(glm::vec3(p1, p2, p3));
                    
                }else if(t == Scalar){
                    data->scalar_data[f_it++]->push_back(p1);
                }
            }
            
            processed += 1;
        }
    }else{
        std::cout << "Ooops .. failed to allocate/open " << path << "\n";
    }
}

void ThreadLoadData(ParserData *owner,
                    std::string head,
                    std::string ext,
                    unsigned int index,
                    std::vector<Types> types)
{
    std::string file(head);
    std::string ct = toString<unsigned int>(index);
    file += ct;
    file += ext;
    std::cout << "Loading " << file << std::endl;
    LoadData(file.c_str(), owner, types);
}


Parser_v2 * Parser_v2_new(const char *format){
    Parser_v2 *parser = new Parser_v2;
    parser->blob.hasScalar = 0;
    parser->blob.hasVector = 0;
    std::string f = format;
    size_t len = f.size();
    int it = 0;
    while(len > 0){
        char chr = f[it++];
        if(chr == 'v' || chr == 'V'){
            parser->blob.types.push_back(Vector);
            parser->blob.hasVector += 1;
        }else if(chr == 's' || chr == 'S'){
            parser->blob.types.push_back(Scalar);
            parser->blob.hasScalar += 1;
        }
        
        len--;
    }
    
    return parser;
}


void Parser_v2_load_single_file(Parser_v2 *parser, const char *name){
    Assert(parser);
    
    ParserData *data = new ParserData;
    if(parser->blob.hasScalar){
        data->scalar_data.resize(parser->blob.hasScalar);
        for(int i = 0; i < parser->blob.hasScalar; i += 1){
            data->scalar_data[i] = new std::vector<float>();
        }
    }
    
    if(parser->blob.hasVector){
        data->vector_data.resize(parser->blob.hasVector);
        for(int i = 0; i < parser->blob.hasVector; i += 1){
            data->vector_data[i] = new std::vector<glm::vec3>();
        }
    }
    
    Timed("Reading file", LoadData(name, data, parser->blob.types));
    
    parser->blob.activeFrame = parser->blob.frames.size();
    parser->blob.frames.push_back(data);
}

void Parser_v2_load_multiple_filesv(Parser_v2 *parser, const char *head,
                                    const char *ext, std::vector<int> steps,
                                    int maxthreads)
{
    Assert(parser);
    int threadPack = maxthreads > 0 ? maxthreads : 1;
    int setId = 0;
    int missing = steps.size();
    unsigned int threadIt = 0;
    int vecId = 0;
    parser->blob.frames.resize(missing);
    DelayedThread *threads = new DelayedThread[threadPack];
    while(missing > 0){
        std::string packName("FileSet load ");
        packName += toString<int>(++setId);
        int packStart = threadIt;
        int toLoad = threadPack < missing ? threadPack : missing;
        packName += "("; packName += toString<int>(packStart);
        packName += ","; packName += toString<int>(packStart+toLoad);
        packName += ")";
        clock_t start = clock();
        std::cout << ">> Running: " << packName << std::endl;
        for(int i = 0; i < toLoad; i += 1){
            int fileIndex = steps[vecId++];
            if(threads[i].active){
                std::cout << "Oops... tried to start already running thread\n";
                getchar();
            }else{
                ParserData *data = new ParserData;
                if(parser->blob.hasScalar){
                    data->scalar_data.resize(parser->blob.hasScalar);
                    for(int i = 0; i < parser->blob.hasScalar; i += 1){
                        data->scalar_data[i] = new std::vector<float>();
                    }
                }
                
                if(parser->blob.hasVector){
                    data->vector_data.resize(parser->blob.hasVector);
                    for(int i = 0; i < parser->blob.hasVector; i += 1){
                        data->vector_data[i] = new std::vector<glm::vec3>();
                    }
                }
                data->index = fileIndex;
                parser->blob.frames[threadIt] = data;
                threads[i].launch(ThreadLoadData,
                                  parser->blob.frames[threadIt],
                                  head, ext, data->index,
                                  parser->blob.types);
                
                threadIt += 1;
            }
        }
        
        bool done = false;
        while(!done){
            done = true;
            for(int i = 0; i < toLoad; i += 1){
                if(threads[i].active) done = false;
            }
            if(!done)
                Sleep(100);
        }
        
        missing -= toLoad;
        clock_t end = clock();
        double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        std::cout << "Took >> " << cpu_time_used << " s" << std::endl; 
    }
}

void Parser_v2_load_multiple_files(Parser_v2 * parser, const char *head,
                                   const char *ext,
                                   int start, int end, int maxthreads)
{
    Assert(parser);
    int threadPack = maxthreads > 0 ? maxthreads : 1;
    int setId = 0;
    int missing = end - start;
    unsigned int threadIt = 0;
    unsigned int fileIndex = start;
    parser->blob.frames.resize(missing);
    DelayedThread *threads = new DelayedThread[threadPack];
    
    while(missing > 0){
        std::string packName("FileSet load ");
        packName += toString<int>(++setId);
        int packStart = threadIt;
        int toLoad = threadPack < missing ? threadPack : missing;
        packName += "("; packName += toString<int>(packStart);
        packName += ","; packName += toString<int>(packStart+toLoad);
        packName += ")";
        
        clock_t start = clock();
        std::cout << ">> Running: " << packName << std::endl;
        for(int i = 0; i < toLoad; i += 1){
            if(threads[i].active){
                std::cout << "Oops... tried to start already running thread\n";
                getchar();
            }else{
                ParserData *data = new ParserData;
                if(parser->blob.hasScalar){
                    data->scalar_data.resize(parser->blob.hasScalar);
                    for(int i = 0; i < parser->blob.hasScalar; i += 1){
                        data->scalar_data[i] = new std::vector<float>();
                    }
                }
                
                if(parser->blob.hasVector){
                    data->vector_data.resize(parser->blob.hasVector);
                    for(int i = 0; i < parser->blob.hasVector; i += 1){
                        data->vector_data[i] = new std::vector<glm::vec3>();
                    }
                }
                data->index = fileIndex;
                parser->blob.frames[threadIt] = data;
                threads[i].launch(ThreadLoadData,
                                  parser->blob.frames[threadIt],
                                  head, ext, data->index,
                                  parser->blob.types);
                
                threadIt += 1;
                fileIndex += 1;
            }
        }
        
        bool done = false;
        while(!done){
            done = true;
            for(int i = 0; i < toLoad; i += 1){
                if(threads[i].active) done = false;
            }
            if(!done)
                Sleep(100);
        }
        
        missing -= toLoad;
        
        clock_t end = clock();
        double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        std::cout << "Took >> " << cpu_time_used << " s" << std::endl; 
    }
}

std::vector<glm::vec3> * Parser_v2_get_vector_ptr(Parser_v2 *parser, int idx){
    int active = parser->blob.activeFrame;
    Assert(idx < parser->blob.frames[active]->vector_data.size());
    return parser->blob.frames[active]->vector_data[idx];
}

std::vector<float> * Parser_v2_get_scalar_ptr(Parser_v2 *parser, int idx){
    int active = parser->blob.activeFrame;
    Assert(idx < parser->blob.frames[active]->scalar_data.size());
    return parser->blob.frames[active]->scalar_data[idx];
}


std::vector<glm::vec3> * Parser_v2_get_vector_ptr(Parser_v2 *parser, int frame, int i){
    Assert(frame < parser->blob.frames.size());
    Assert(i < parser->blob.frames[frame]->vector_data.size());
    return parser->blob.frames[frame]->vector_data[i];
}

std::vector<float> * Parser_v2_get_scalar_ptr(Parser_v2 *parser, int frame, int i){
    Assert(frame < parser->blob.frames.size());
    Assert(i < parser->blob.frames[frame]->scalar_data.size());
    return parser->blob.frames[frame]->scalar_data[i];
}

glm::vec3 * Parser_v2_get_raw_vector_ptr(Parser_v2 *parser, int frame, 
                                         int i, size_t *count)
{
    Assert(frame < parser->blob.frames.size());
    Assert(i < parser->blob.frames[frame]->vector_data.size());
    *count = parser->blob.frames[frame]->vector_data[i]->size();
    return &(parser->blob.frames[frame]->vector_data[i]->operator[](0));
}

float * Parser_v2_get_raw_scalar_ptr(Parser_v2 *parser, int frame,
                                     int i, size_t *count)
{
    Assert(frame < parser->blob.frames.size());
    Assert(i < parser->blob.frames[frame]->scalar_data.size());
    *count = parser->blob.frames[frame]->scalar_data[i]->size();
    return &(parser->blob.frames[frame]->scalar_data[i]->operator[](0));
}
