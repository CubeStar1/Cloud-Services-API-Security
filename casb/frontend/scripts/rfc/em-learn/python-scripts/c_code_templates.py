C_INCLUDES_TEMPLATE = """#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#include <eml_trees.h>
#include "include/{service_model_c_name}.h"
#include "include/{activity_model_c_name}.h"
#include "include/feature_engineering.h"

"""

C_MAIN_TEMPLATE = """\

static int16_t features[MAX_C_FEATURES]; 

static const char* s_host;
static const char* s_url;
static const char* s_method;
static const char* s_origin;
static const char* s_req_content_type;
static const char* s_res_content_type;
static const char* s_referer;
static const char* s_accept;

int main() {{
    printf("API Classifier Demo\\\\n\\\\n"); 
    int service_idx;
    int activity_idx;

{c_test_cases_dynamic}

    printf("\\\\nEnd of demo.\\\\n");
    return 0;
}}
"""

C_FEATURE_ENGINEERING_H_TEMPLATE = """\
#ifndef FEATURE_ENGINEERING_H
#define FEATURE_ENGINEERING_H

#include <stdint.h> 
#include <string.h> 
#include <ctype.h>  

#define MAX_C_FEATURES {MAX_C_FEATURES_DEF}
#define C_HASH_TABLE_SIZE {C_HASH_TABLE_SIZE_DEF}
#define C_MAX_BUCKET_SIZE {C_MAX_BUCKET_SIZE_DEF}

typedef struct {{
    const char* term;
    int16_t feature_index; 
}} FeatureEntryC;

static const FeatureEntryC FEATURE_TABLE_C[] = {{
{feature_table_entries_dynamic}
}};

#define NUM_C_VOCAB_TERMS (sizeof(FEATURE_TABLE_C) / sizeof(FeatureEntryC))

typedef struct {{
    int16_t indices[C_MAX_BUCKET_SIZE]; // Store indices into FEATURE_TABLE_C
    int8_t count;                     // Number of valid indices in this bucket
}} HashBucketC;

static const HashBucketC HASH_BUCKETS_C[C_HASH_TABLE_SIZE] = {{
{hash_bucket_entries_dynamic}
}};

static inline uint32_t fnv1a_hash_c(const char* str) {{
    uint32_t hash = 2166136261u;
    while (*str) {{
        hash ^= (unsigned char)*str;
        hash *= 16777619u;
        str++;
    }}
    return hash; 
}}

static inline int16_t find_feature_c(const char* term) {{
    uint32_t hash = fnv1a_hash_c(term) % C_HASH_TABLE_SIZE;
    const HashBucketC* bucket = &HASH_BUCKETS_C[hash];
    for (int i = 0; i < bucket->count; i++) {{
        int16_t table_idx = bucket->indices[i];
        // Bounds check for safety, though ideally data generation ensures validity
        if (table_idx >= 0 && table_idx < (int16_t)NUM_C_VOCAB_TERMS) {{
            if (strcmp(FEATURE_TABLE_C[table_idx].term, term) == 0) {{
                return FEATURE_TABLE_C[table_idx].feature_index;
            }}
        }}
    }}
    return -1; 
}}

static inline void extract_features_from_strings(
    const char* s_host, 
    const char* s_url, 
    const char* s_method, 
    const char* s_origin, 
    const char* s_req_content_type, 
    const char* s_res_content_type, 
    const char* s_referer, 
    const char* s_accept, 
    int16_t features[MAX_C_FEATURES]) {{

    // 1. Initialize all features to 0
    for (int i = 0; i < MAX_C_FEATURES; i++) {{
        features[i] = 0;
    }}

    // 2. Define the array of input strings
    const char* inputs[] = {{
        s_host ? s_host : "",
        s_url ? s_url : "",
        s_method ? s_method : "",
        s_origin ? s_origin : "",
        s_req_content_type ? s_req_content_type : "",
        s_res_content_type ? s_res_content_type : "",
        s_referer ? s_referer : "",
        s_accept ? s_accept : ""
    }};

    char token_buffer[1024]; 
    int token_len;

    // 3. Process each input field
    for (int i = 0; i < 8; i++) {{ 
        const char* p = inputs[i];
        while (*p) {{
            while (*p && !isalnum((unsigned char)*p)) {{
                p++;
            }}
            if (!*p) break; 

            // const char* token_start = p; 
            token_len = 0;
            while (*p && isalnum((unsigned char)*p) && token_len < (sizeof(token_buffer) - 1)) {{
                token_buffer[token_len++] = tolower((unsigned char)*p); 
                p++;
            }}
            token_buffer[token_len] = '\\0'; 

            // Process token if length >= 2 (matches r"(?u)\\b\\w\\w+\\b")
            if (token_len >= 2) {{
                int16_t feature_idx = find_feature_c(token_buffer);
                if (feature_idx >= 0 && feature_idx < MAX_C_FEATURES) {{ 
                    features[feature_idx] = 1; 
                }}
            }}
        }}
    }}
}}

#endif // FEATURE_ENGINEERING_H
""" 