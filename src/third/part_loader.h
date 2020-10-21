#pragma once
#include <geometry.h>
#include <cutil.h>

/*
* Loads a particle description file. Currently this is only to support
* Bubbles particles output, which is basically a CSV file. Ok look, there is
* no point in performing multiple output load as this will ray trace things
* and will be very slow, so let's just focus on loading a single description.
*/

/*
* Output parsed data is GPU ready.
*/

typedef struct{
    vec3f **vector_data;
    Float **scalar_data;
    int nvectors;
    int nscalars;
    int count;
}PParsedData;

typedef struct{
    std::string format;
    PParsedData *data;
    int state;
}PParser;

/*
* Initializes a instance of the particle parser with the given format.
*/
__host__ void PParserInitialize(PParser *parser, const char *format);

/*
* Attempt to parse a single output from Bubbles.
*/
__host__ bool PParserParse(PParser *parser, const char *filename);

/*
* Cleanup loaded memory
*/
__host__ void PParserCleanup(PParser *parser);

/*
* Get pointer to a vector data parsed located at 'index'.
*/
__host__ vec3f *PParserGetVectorPtr(PParser *parser, int index, int *count);

/*
* Get pointer to a scalar data parsed located at 'index'.
*/
__host__ Float *PParserGetScalarPtr(PParser *parser, int index, int *count);