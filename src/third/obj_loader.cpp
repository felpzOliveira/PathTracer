#include <obj_loader.h>
#include <fstream>
#include <cutil.h>
#include <sstream>

#define IS_SPACE(x) (((x) == ' ') || ((x) == '\t'))
#define IS_DIGIT(x) (static_cast<unsigned int>((x) - '0') < static_cast<unsigned int>(10))
#define IS_NEW_LINE(x) (((x) == '\r') || ((x) == '\n') || ((x) == '\0'))

/*
* Heavily based on tiny_obj_loader. I'm basically just making sure
* multiple meshes are outputed correctly and MTL files get correctly processed.
*/

struct vertex_index_t {
    int v_idx, vt_idx, vn_idx;
    vertex_index_t() : v_idx(-1), vt_idx(-1), vn_idx(-1) {}
    explicit vertex_index_t(int idx) : v_idx(idx), vt_idx(idx), vn_idx(idx) {}
    vertex_index_t(int vidx, int vtidx, int vnidx)
        : v_idx(vidx), vt_idx(vtidx), vn_idx(vnidx) {}
};

static inline bool fixIndex(int idx, int n, int *ret) {
    if (!ret) {
        return false;
    }
    
    if (idx > 0) {
        (*ret) = idx - 1;
        return true;
    }
    
    if (idx == 0) {
        // zero is not allowed according to the spec.
        return false;
    }
    
    if (idx < 0) {
        (*ret) = n + idx;  // negative value = relative
        return true;
    }
    
    return false;  // never reach here.
}

static bool parseTriple(const char **token, int vsize, int vnsize, int vtsize,
                        vertex_index_t *ret)
{
    if (!ret) {
        return false;
    }
    
    vertex_index_t vi(-1);
    
    if (!fixIndex(atoi((*token)), vsize, &(vi.v_idx))) {
        return false;
    }
    
    (*token) += strcspn((*token), "/ \t\r");
    if ((*token)[0] != '/') {
        (*ret) = vi;
        return true;
    }
    (*token)++;
    
    // i//k
    if ((*token)[0] == '/') {
        (*token)++;
        if (!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))) {
            return false;
        }
        (*token) += strcspn((*token), "/ \t\r");
        (*ret) = vi;
        return true;
    }
    
    // i/j/k or i/j
    if (!fixIndex(atoi((*token)), vtsize, &(vi.vt_idx))) {
        return false;
    }
    
    (*token) += strcspn((*token), "/ \t\r");
    if ((*token)[0] != '/') {
        (*ret) = vi;
        return true;
    }
    
    // i/j/k
    (*token)++;  // skip '/'
    if (!fixIndex(atoi((*token)), vnsize, &(vi.vn_idx))) {
        return false;
    }
    (*token) += strcspn((*token), "/ \t\r");
    
    (*ret) = vi;
    
    return true;
}

static bool tryParseDouble(const char *s, const char *s_end, Float *result) {
    if (s >= s_end) {
        return false;
    }
    
    double mantissa = 0.0;
    // This exponent is base 2 rather than 10.
    // However the exponent we parse is supposed to be one of ten,
    // thus we must take care to convert the exponent/and or the
    // mantissa to a * 2^E, where a is the mantissa and E is the
    // exponent.
    // To get the final double we will use ldexp, it requires the
    // exponent to be in base 2.
    int exponent = 0;
    
    // NOTE: THESE MUST BE DECLARED HERE SINCE WE ARE NOT ALLOWED
    // TO JUMP OVER DEFINITIONS.
    char sign = '+';
    char exp_sign = '+';
    char const *curr = s;
    
    // How many characters were read in a loop.
    int read = 0;
    // Tells whether a loop terminated due to reaching s_end.
    bool end_not_reached = false;
    
    /*
            BEGIN PARSING.
    */
    
    // Find out what sign we've got.
    if (*curr == '+' || *curr == '-') {
        sign = *curr;
        curr++;
    } else if (IS_DIGIT(*curr)) { /* Pass through. */
    } else {
        goto fail;
    }
    
    // Read the integer part.
    end_not_reached = (curr != s_end);
    while (end_not_reached && IS_DIGIT(*curr)) {
        mantissa *= 10;
        mantissa += static_cast<int>(*curr - 0x30);
        curr++;
        read++;
        end_not_reached = (curr != s_end);
    }
    
    // We must make sure we actually got something.
    if (read == 0) goto fail;
    // We allow numbers of form "#", "###" etc.
    if (!end_not_reached) goto assemble;
    
    // Read the decimal part.
    if (*curr == '.') {
        curr++;
        read = 1;
        end_not_reached = (curr != s_end);
        while (end_not_reached && IS_DIGIT(*curr)) {
            static const double pow_lut[] = {
                1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
            };
            const int lut_entries = sizeof pow_lut / sizeof pow_lut[0];
            
            // NOTE: Don't use powf here, it will absolutely murder precision.
            mantissa += static_cast<int>(*curr - 0x30) *
                (read < lut_entries ? pow_lut[read] : std::pow(10.0, -read));
            read++;
            curr++;
            end_not_reached = (curr != s_end);
        }
    } else if (*curr == 'e' || *curr == 'E') {
    } else {
        goto assemble;
    }
    
    if (!end_not_reached) goto assemble;
    
    // Read the exponent part.
    if (*curr == 'e' || *curr == 'E') {
        curr++;
        // Figure out if a sign is present and if it is.
        end_not_reached = (curr != s_end);
        if (end_not_reached && (*curr == '+' || *curr == '-')) {
            exp_sign = *curr;
            curr++;
        } else if (IS_DIGIT(*curr)) { /* Pass through. */
        } else {
            // Empty E is not allowed.
            goto fail;
        }
        
        read = 0;
        end_not_reached = (curr != s_end);
        while (end_not_reached && IS_DIGIT(*curr)) {
            exponent *= 10;
            exponent += static_cast<int>(*curr - 0x30);
            curr++;
            read++;
            end_not_reached = (curr != s_end);
        }
        exponent *= (exp_sign == '+' ? 1 : -1);
        if (read == 0) goto fail;
    }
    
    assemble:
    *result = (sign == '+' ? 1 : -1) *
        (exponent ? std::ldexp(mantissa * std::pow(5.0, exponent), exponent)
         : mantissa);
    return true;
    fail:
    return false;
}

static std::istream &GetLine(std::istream &is, std::string &t){
    t.clear();
    std::istream::sentry se(is, true);
    std::streambuf *sb = is.rdbuf();
    if(se){
        for(;;){
            int c = sb->sbumpc();
            switch(c){
                case '\n': return is;
                case '\r': if(sb->sgetc() == '\n') sb->sbumpc(); return is;
                case EOF: if(t.empty()) is.setstate(std::ios::eofbit); return is;
                default: t += static_cast<char>(c);
            }
        }
    }
    
    return is;
}

static inline Float ParseFloat(const char **token){
    (*token) += strspn((*token), " \t");
    const char *end = (*token) + strcspn((*token), " \t\r");
    Float val = 0;
    tryParseDouble((*token), end, &val);
    Float f = static_cast<Float>(val);
    (*token) = end;
    return f;
}

static inline void ParseV3(vec3f *v, const char **token){
    *v = vec3f(ParseFloat(token), ParseFloat(token), ParseFloat(token));
}

static inline void ParseV2(vec2f *v, const char **token){
    *v = vec2f(ParseFloat(token), ParseFloat(token));
}

__host__ std::vector<ParsedMesh*> *LoadObj(const char *path, std::vector<MeshMtl> *mtls){
    ParsedMesh *currentMesh = nullptr;
    std::vector<vec3f> v, vn;
    std::vector<vec2f> vt;
    std::vector<vertex_index_t> indexes;
    std::vector<ParsedMesh*> *meshes = new std::vector<ParsedMesh *>();
    
    vertex_index_t face[4];
    int facen = 0;
    
    int *picked = nullptr;
    int pickedSize = 0;
    printf("Attempting to parse %s\n", path);
    
    std::ifstream ifs(path);
    if(!ifs){
        printf("Could not open file %s\n", path);
        return meshes;
    }
    
    std::string linebuf;
    
    int making_mesh = 0;
    std::string currentMtlFile;
    std::string currentMaterialName;
    int matNameCounter = 0;
    
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
        
        //if we just ended a mesh
        if(token[0] != 'f' && making_mesh){
            making_mesh = 0;
            std::vector<Point3f> p2;
            int c = 0;
            printf("Mesh with # indices: %d [ ", (int)indexes.size());
            if(!picked){
                picked = new int[v.size()];
                pickedSize = v.size();
            }else if(pickedSize < v.size()){
                delete[] picked;
                picked = new int[v.size()];
                pickedSize = v.size();
            }
            
            for(int i = 0; i < v.size(); i++) picked[i] = -1;
            for(int i = 0; i < indexes.size(); i++){
                int which = indexes[i].v_idx;
                if(picked[which] == -1){
                    picked[which] = c;
                    vec3f p = v[which];
                    p2.push_back(Point3f(p.x, p.y, p.z));
                    c++;
                }
            }
            
            currentMesh->p = cudaAllocateVx(Point3f, p2.size());
            currentMesh->indices = cudaAllocateVx(int, indexes.size());
            currentMesh->nTriangles = indexes.size() / 3;
            currentMesh->nVertices = p2.size();
            memcpy(currentMesh->p, p2.data(), p2.size() * sizeof(Point3f));
            for(int i = 0; i < indexes.size(); i++){
                int idx = indexes[i].v_idx;
                currentMesh->indices[i] = picked[idx];
            }
            
            for(int i = 0; i < 6; i++){
                printf("%d ", currentMesh->indices[i]);
            }
            
            printf("]\n");
        }
        
        if(token[0] == 'o' && IS_SPACE((token[1]))){
            //TODO: grab name
        }
        
        if(token[0] == 'v' && IS_SPACE((token[1]))){
            token += 2;
            vec3f vertex;
            ParseV3(&vertex, &token);
            v.push_back(vertex);
            continue;
        }
        
        if(token[0] == 'v' && token[1] == 'n' && IS_SPACE((token[2]))){
            token += 3;
            vec3f normal;
            ParseV3(&normal, &token);
            vn.push_back(normal);
            continue;
        }
        
        if(token[0] == 'v' && token[1] == 't' && IS_SPACE((token[2]))){
            token += 3;
            vec2f uv;
            ParseV2(&uv, &token);
            vt.push_back(uv);
            continue;
        }
        
        if((0 == strncmp(token, "mtllib", 6)) && IS_SPACE((token[6]))){
            token += 7;
            currentMtlFile = std::string(token);
            printf("Materials to lookup here: %s\n", currentMtlFile.c_str());
            continue;
        }
        
        if((0 == strncmp(token, "usemtl", 6)) && IS_SPACE((token[6]))){
            token += 7;
            std::stringstream ss;
            ss << token;
            currentMaterialName = ss.str();
            matNameCounter ++;
            printf("Found material %s\n", currentMaterialName.c_str());
            continue;
        }
        
        if(token[0] == 'f' && IS_SPACE((token[1]))){
            //NOTE: entering here means we discovered a new mesh
            if(!making_mesh){
                MeshMtl mtl;
                currentMesh = cudaAllocateVx(ParsedMesh, 1);
                currentMesh->nVertices = 0;
                currentMesh->nTriangles = 0;
                meshes->push_back(currentMesh);
                mtl.file = currentMtlFile;
                mtl.name = currentMaterialName;
                mtls->push_back(mtl);
                indexes.clear();
                making_mesh = 1;
            }
            
            token += 2;
            token += strspn(token, " \t");
            facen = 0;
            while(!IS_NEW_LINE(token[0])){
                vertex_index_t vi;
                if (!parseTriple(&token, static_cast<int>(v.size()),
                                 static_cast<int>(vn.size()),
                                 static_cast<int>(vt.size()), &vi)) 
                {
                    printf("Failed parsing face\n");
                    break;
                }
                
                size_t n = strspn(token, " \t\r");
                token += n;
                if(facen >= 4){
                    printf("Not a triangle or quad face\n");
                    exit(0);
                }
                
                face[facen++] = vi;
            }
            
            if(facen == 3){
                indexes.push_back(face[0]); indexes.push_back(face[1]);
                indexes.push_back(face[2]);
            }else if(facen == 4){
                indexes.push_back(face[0]); indexes.push_back(face[1]);
                indexes.push_back(face[2]); indexes.push_back(face[0]); 
                indexes.push_back(face[2]); indexes.push_back(face[3]);
            }
            
            continue;
        }
    }
    
    if(making_mesh){
        std::vector<Point3f> p2;
        int c = 0;
        printf("Mesh with # indices: %d [", (int)indexes.size());
        if(!picked){
            picked = new int[v.size()];
            pickedSize = v.size();
        }else if(pickedSize < v.size()){
            delete[] picked;
            picked = new int[v.size()];
            pickedSize = v.size();
        }
        
        for(int i = 0; i < v.size(); i++) picked[i] = -1;
        for(int i = 0; i < indexes.size(); i++){
            int which = indexes[i].v_idx;
            if(picked[which] == -1){
                picked[which] = c;
                vec3f p = v[which];
                p2.push_back(Point3f(p.x, p.y, p.z));
                c++;
            }
        }
        
        currentMesh->p = cudaAllocateVx(Point3f, p2.size());
        currentMesh->indices = cudaAllocateVx(int, indexes.size());
        currentMesh->nTriangles = indexes.size() / 3;
        currentMesh->nVertices = p2.size();
        memcpy(currentMesh->p, p2.data(), p2.size() * sizeof(Point3f));
        for(int i = 0; i < indexes.size(); i++){
            int idx = indexes[i].v_idx;
            currentMesh->indices[i] = picked[idx];
        }
        
        for(int i = 0; i < 6; i++){
            printf("%d ", currentMesh->indices[i]);
        }
        
        printf("]\n");
    }
    
    if(picked) free(picked);
    
    clock_t end = clock();
    
    double time_taken = to_cpu_time(start, end);
    printf("Took %g, vertices [%d] normals [%d] uvs [%d]\n", time_taken, (int)v.size(),
           (int)vn.size(), (int)vt.size());
    
    return meshes;
}