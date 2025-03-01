from beartype.typing import Generator, Iterable
from jax import (
    numpy as jnp,
    tree as jt,
)

from .vmapped import VmappedT


def stack[T](trees: Iterable[T]) -> VmappedT[T, " _"]:
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jt.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def unstack[T](tree: VmappedT[T, " _"]) -> Generator[T]:
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jt.flatten(tree)
    n_trees = leaves[0].shape[0]
    for i in range(n_trees):
        new_leaves = [leaf[i] for leaf in leaves]
        yield treedef.unflatten(new_leaves)
