#include "linked_list.c"

void main()
{
    node *t;
    init_list();
    ordered_insert(9);
    ordered_insert(3);
    ordered_insert(5);
    ordered_insert(1);
    ordered_insert(7);
    t = head;
    while(t != tail){
        printf("%3d",t->key);
        t = t->next;
    }
    printf("\n");
    ordered_insert(6);
    t = head;
    while(t != tail){
        printf("%3d",t->key);
        t = t->next;
    }
    printf("\n");
    delete_node(5);
    t=head;
    while(t != tail){
        printf("%3d",t->key);
        t = t->next;
    }
    printf("\n");
    delete_all();
}
