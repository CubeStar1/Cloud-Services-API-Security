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

C_MAIN_TEMPLATE = """int main() {{
    printf("emlearn RFC Inference with C Feature Extraction\\n");
    static int16_t features[MAX_C_FEATURES];
    int32_t service_idx, activity_idx;

    const char *s_host, *s_url, *s_method, *s_origin, *s_req_content_type, *s_res_content_type, *s_referer, *s_accept;

{c_test_cases_dynamic}
    return EXIT_SUCCESS;
}}
"""

C_FEATURE_ENGINEERING_H_TEMPLATE = """#ifndef FEATURE_ENGINEERING_H
#define FEATURE_ENGINEERING_H

#include <stdio.h> 
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h> 

#define MAX_C_FEATURES {MAX_C_FEATURES_DEF}
#define C_HASH_TABLE_SIZE {C_HASH_TABLE_SIZE_DEF}
#define C_MAX_BUCKET_SIZE {C_MAX_BUCKET_SIZE_DEF}

typedef struct {{ const char* term; int feature_index; }} FeatureEntry;
static const FeatureEntry FEATURE_TABLE[] = {{
{feature_table_entries_dynamic}
}};
#define NUM_VOCAB_ENTRIES (sizeof(FEATURE_TABLE) / sizeof(FEATURE_TABLE[0]))

typedef struct {{ int indices[C_MAX_BUCKET_SIZE]; int count; }} HashBucket;
static const HashBucket HASH_BUCKETS[C_HASH_TABLE_SIZE] = {{
{hash_bucket_entries_dynamic}
}};

static inline unsigned int fnv1a_hash_c(const char* str) {{
    unsigned int hash = 2166136261u;
    if (!str) return hash % C_HASH_TABLE_SIZE; 
    while (*str) {{
        hash ^= (unsigned char)*str;
        hash *= 16777619;
        str++;
    }}
    return hash % C_HASH_TABLE_SIZE;
}}

static inline int find_feature_index(const char* term) {{
    if (!term) return -1;
    unsigned int hash = fnv1a_hash_c(term);
    const HashBucket* bucket = &HASH_BUCKETS[hash];
    for (int i = 0; i < bucket->count; i++) {{
        int table_idx = bucket->indices[i];
        if (table_idx >= 0 && table_idx < NUM_VOCAB_ENTRIES) {{
             if (strcmp(FEATURE_TABLE[table_idx].term, term) == 0) {{
                return FEATURE_TABLE[table_idx].feature_index;
            }}
        }}
    }}
    return -1;
}}

static inline void extract_features_from_strings(
    const char* s_host, const char* s_url, const char* s_method,
    const char* s_origin, const char* s_req_content_type, const char* s_res_content_type,
    const char* s_referer, const char* s_accept,
    int16_t features[]
) {{
    for (int i = 0; i < MAX_C_FEATURES; i++) features[i] = 0;

    const char* inputs[] = {{ s_host, s_url, s_method, s_origin, s_req_content_type, s_res_content_type, s_referer, s_accept }};
    char token_buffer[1024];

    for (int i = 0; i < 8; ++i) {{
        const char* current_input_str = inputs[i] ? inputs[i] : "";
        const char* p = current_input_str;
        while (*p) {{
            while (*p && !isalnum((unsigned char)*p)) p++;
            if (!*p) break;
            
            int token_len = 0;
            while (*p && isalnum((unsigned char)*p) && token_len < 1023) {{
                token_buffer[token_len++] = tolower((unsigned char)*p);
                p++;
            }}
            token_buffer[token_len] = '\\0';

            if (token_len >= 2) {{
                int feature_idx = find_feature_index(token_buffer);
                if (feature_idx != -1 && feature_idx < MAX_C_FEATURES) {{
                    features[feature_idx] = 1;
                }}
            }}
        }}
    }}
}}

#endif
""" 