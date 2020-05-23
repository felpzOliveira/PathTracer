#include <mtl.h>
#include <fstream>
#include <sstream>

enum Token{
    NewMaterial, Variable, None
};

typedef struct{
    Token token;
    std::string name;
    bool makenew;
    std::vector<std::string> *values;
}MTLToken;

static const int mtlToParseCount = 11;
static const char *mtlVars[mtlToParseCount] = {
    "Ka", "Kd", "Ks", "Tf", "Ns", "illum",
    "map_Ka", "map_Kd", "map_Ks", "d", "Ke",
};


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
        printf("Unkown state for line: %s\n", line.c_str());
    }
    
    return mtlToken;
}

bool MTLParse(const char *path, std::vector<MTL *> *materials){
    if(!materials){
        printf("Invalid materials pointer for MTL parsing\n");
        return false;
    }
    
    std::string line;
    std::ifstream infile(path);
    int in_context = 0;
    if(infile.is_open()){
        while(std::getline(infile, line)){
            MTLToken mtlToken = MTLParseLine(line, &in_context);
            if(mtlToken.token != Token::None){
                printf("Name: %s", mtlToken.name.c_str());
                if(mtlToken.values){
                    printf(" [");
                    for(int i = 0; i < mtlToken.values->size(); i++){
                        std::string val = mtlToken.values->at(i);
                        if(i < mtlToken.values->size()-1)
                            printf("%s ", val.c_str());
                        else 
                            printf("%s", val.c_str());
                    }
                    
                    printf("]");
                }
                
                printf("\n");
                
                delete mtlToken.values;
            }
        }
    }else{
        printf("Could not open file: %s\n", path);
    }
    
    return false;
}