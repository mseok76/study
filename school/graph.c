
#include "header/graph_list.c"
#include "header/graph_matrix.c"

void main()
{
    int V, E;
    // for adjacency matrix
    // input_adjmatrix(GM, &V, &E);
    // print_adjmatrix(GM, V);
    // for adjacency list
    input_adjlist(GL, &V, &E);
    print_adjlist(GL, V);
}
