#include <stdio.h>
#include "personalization.h"

int main(void) {
    linear_head_t linear_head;
    prototype_head_t proto_head;
    float example_embedding[EMBEDDING_DIM] = {0};

    linear_head_init(&linear_head);
    prototype_head_reset(&proto_head);
    prototype_head_update(&proto_head, example_embedding, 0);

    int pred_linear = linear_head_predict(&linear_head, example_embedding);
    int pred_proto = prototype_head_predict(&proto_head, example_embedding);

    printf("pre_adapt_linear=%d post_adapt_proto=%d\n", pred_linear, pred_proto);
    return 0;
}
