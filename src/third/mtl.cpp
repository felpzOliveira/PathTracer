#include <mtl.h>
#include <fstream>
#include <sstream>

#define IS_DIGIT(x) (static_cast<unsigned int>((x) - '0') < static_cast<unsigned int>(10))

enum Token{
    NewMaterial, Variable, None
};

typedef struct{
    Token token;
    std::string name;
    bool makenew;
    std::vector<std::string> *values;
}MTLToken;

static const int mtlToParseCount = 12;
static const char *mtlVars[mtlToParseCount] = {
    "Ka", "Kd", "Ks", "Tf", "Ns", "illum",
    "map_Ka", "map_Kd", "map_Ks", "d", "Ke", "Ni",
};
static const int mtlVarsNumeric[mtlToParseCount] = {
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 1, 1,
};

static bool IsNumeric(const char *property){
    bool rv = false;
    std::string target(property);
    for(int i = 0; i < mtlToParseCount; i++){
        if(target == std::string(mtlVars[i])) 
            return mtlVarsNumeric[i];
    }
    
    return rv;
}

bool ParseDouble(const char *s, const char *s_end, Float *result){
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

static bool StringStartsWith(const std::string &content, 
                             const std::string &search)
{
    return content.rfind(search, 0) == 0;
}

static int GetNextEmptyPosition(const std::string &content, int pos, int *is_terminator){
    while((content[pos] != ' ' && content[pos] != '\n' && content[pos] != '\r')
          && pos < content.size()) pos++;
    
    *is_terminator = 0;
    if(content[pos] == '\n' || content[pos] == '\r') *is_terminator = 1;
    return pos;
}

static int GetNextPosition(const std::string &content, int pos){
    while((content[pos] == ' ' || content[pos] == '\n' || content[pos] == '\r')
          && pos < content.size()) pos++;
    return pos;
}

static std::string GetNextString(const std::string &content, int pos, bool *done){
    std::string resp;
    *done = true;
    pos = GetNextPosition(content, pos);
    if(pos < content.size()){
        int terminator = 0;
        int start = pos, end = pos;
        end = GetNextEmptyPosition(content, pos, &terminator);
        resp = content.substr(start, end - start);
        *done = terminator != 0;
    }
    
    return resp;
}

static MTLToken MTLParseLine(const std::string &line, int *is_in_context){
    MTLToken mtlToken;
    mtlToken.makenew = false;
    mtlToken.token = Token::None;
    mtlToken.values = nullptr;
    bool done = false;
    if(line.size() > 0){
        if(StringStartsWith(line, "newmtl")){
            mtlToken.token = Token::NewMaterial;
            mtlToken.makenew = true;
            mtlToken.name = GetNextString(line, 6, &done);
            *is_in_context = 1;
        }else if(*is_in_context){
            for(int i = 0; i < mtlToParseCount; i++){
                std::string var(mtlVars[i]);
                int curr = 0;
                
                if(StringStartsWith(line, var)){
                    curr += var.size();
                    mtlToken.token = Token::Variable;
                    mtlToken.name = std::string(var);
                    mtlToken.values = new std::vector<std::string>();
                    while(!done){
                        std::string val = GetNextString(line, curr, &done);
                        if(val.size() > 0){
                            mtlToken.values->push_back(val);
                            curr += val.size()+1;
                        }
                    }
                    
                    return mtlToken;
                }
            }
            
            printf("Could not find token for line: %s\n", line.c_str());
            *is_in_context = 0;
        }else if(StringStartsWith(line, "#") || line.size() < 1){
            //ignore
        }else{
            //printf("Unkown state for line: %s\n", line.c_str());
            //ignore
        }
    }
    return mtlToken;
}

bool MTLParse(const char *path, std::vector<MTL *> *materials, const char *basedir){
    if(!materials){
        printf("Invalid materials pointer for MTL parsing\n");
        return false;
    }
    
    bool rv = true;
    std::string line;
    std::string filename(basedir);
    filename += std::string(path);
    std::ifstream infile(filename.c_str());
    int in_context = 0;
    if(infile.is_open()){
        MTL *currMaterial = nullptr;
        while(std::getline(infile, line)){
            if(line.size() > 0){ //remove '\n'
                if(line[line.size()-1] == '\n') line.erase(line.size() - 1);
            }
            
            if(line.size() > 0){ //remove '\r'
                if(line[line.size()-1] == '\r') line.erase(line.size() - 1);
            }
            
            // skip empty
            if(line.empty()) continue;
            const char *token = line.c_str();
            token += strspn(token, " \t");
            
            if(token[0] == '#') continue;
            
            line = std::string(token);
            
            if(line.size() > 0){
                MTLToken mtlToken = MTLParseLine(line, &in_context);
                if(mtlToken.token != Token::None){
                    if(mtlToken.token == Token::NewMaterial){
                        currMaterial = new MTL;
                        materials->push_back(currMaterial);
                        currMaterial->mtlname = mtlToken.name;
                        currMaterial->basedir = std::string(basedir);
                    }else{ // variables
                        if(mtlToken.values){
                            std::vector<std::string> data(*(mtlToken.values));
                            currMaterial->data[mtlToken.name] = data;
                            delete mtlToken.values;
                        }else{
                            printf("Empty property %s\n", mtlToken.name.c_str());
                        }
                    }
                }
            }
        }
    }else{
        printf("Could not open file: %s\n", path);
        rv = false;
    }
    
    return rv;
}

std::string MTLGetValue(const char *property, MTL *mtl, bool &found){
    std::string resp;
    found = false;
    if(mtl){
        std::map<std::string, std::vector<std::string>>::iterator it;
        it = mtl->data.find(property);
        if(it != mtl->data.end()){
            found = true;
            std::vector<std::string> data = it->second;
            for(int i = 0; i < data.size(); i+= 1){
                resp += data[i];
                if(i < data.size() - 1)
                    resp += " ";
            }
        }
    }
    
    return resp;
}

Spectrum MTLGetSpectrum(const char *property, MTL *mtl, bool &found){
    Spectrum e(0);
    found = false;
    if(mtl){
        std::map<std::string, std::vector<std::string>>::iterator it;
        it = mtl->data.find(property);
        if(it != mtl->data.end()){ // target exists in this mtl
            found = true;
            std::vector<std::string> data = it->second;
            if(IsNumeric(property)){
                bool done = false;
                int count = 0;
                
                // maximum 3 values in a Spectrum
                while(!done && count < 3 && count < data.size()){
                    std::string val = data[count];
                    const char *ptr = val.c_str();
                    const char *end = ptr + val.size();
                    Float value = 0;
                    done = !ParseDouble(ptr, end, &value);
                    if(!done){
                        e[count++] = value;
                    }
                }
                
            }else{
                printf("Warning: Query for non numeric property (%s)\n", property);
            }
        }
    }
    
    return e;
}

MTL *MTLFindMaterial(const char *name, std::vector<MTL *> *materials){
    int chosen = 0;
    for(int i = 0; i < materials->size(); i++){
        MTL *mtl = materials->at(i);
        if(mtl->mtlname == std::string(name)){
            return materials->at(i);
        }
    }
    
    // have to return something
    printf("Warning: Did not find material (%s)\n", name);
    return materials->at(chosen);
}

bool MTLParseAll(std::vector<MTL *> *materials, std::vector<MeshMtl> *desc,
                 const char *basedir)
{
    bool rv = true;
    if(desc){
        std::string dir(basedir);
        for(int i = 0; i < desc->size() && rv; i++){
            MeshMtl *mtl = &desc->operator[](i);
            rv &= MTLParse(mtl->file.c_str(), materials, basedir);
        }
    }
    
    return rv;
}