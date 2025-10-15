import jaxtyping
from beartype.typing import Generator, Iterable
from jax import (
    numpy as jnp,
    tree as jt,
)

from .vmapped import Vmapped

PyTreeDef = type[jaxtyping.PyTreeDef]  # to make mypy happy


def stack[T](trees: Iterable[T]) -> Vmapped[T, " n"]:
    """Stacks every corresponding leaf on a list of PyTrees.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.

    Parameters
    ----------
    T : type, type parameter
        The type of the PyTrees
    trees : Iterable[T]
        A list of PyTrees, all with the same structure ``T``

    Returns
    -------
    Vmapped[T, "n"]
        A PyTree with the same structure as the input trees, but with the leaves stacked


    Examples
    --------

    >>> a = ({ "l": jnp.array([1, 2]), "r": jnp.array(3) },
    ...      jnp.array([4, 5, 6]))
    >>> b = ({ "l": jnp.array([7, 8]), "r": jnp.array(9) },
    ...      jnp.array([10, 11, 12]))
    >>> c = typinox.tree.stack([a, b])
    >>> d = ({ "l": jnp.array([[1, 2], [7, 8]]), "r": jnp.array([3, 9]) },
    ...      jnp.array([[4, 5, 6], [10, 11, 12]]))
    >>> chex.assert_trees_all_equal(c, d)

    """
    leaves_list = []
    treedef_list: list[PyTreeDef] = []
    for tree in trees:
        leaves, treedef = jt.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def unstack[T](tree: Vmapped[T, " _"]) -> Generator[T]:
    """Unstacks a PyTree of arrays into a list of PyTrees. Inverse of :func:`stack`.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])].

    Useful for turning the output of a vmapped function into normal objects.

    Parameters
    ----------
    T : type, type parameter
        The type of the PyTrees
    tree : Vmapped[T, " _"]
        A PyTree of structure ``T`` with the leaves stacked

    Yields
    ------
    Generator[T]
        A generator that yields PyTrees with the same structure ``T`` as the input tree,
        but with the leaves unstacked

    Examples
    --------

    >>> a = ({ "l": jnp.array([[1, 2], [3, 4]]), "r": jnp.array([5, 6]) },
    ...      jnp.array([7, 8]))
    >>> aa = list(typinox.tree.unstack(a))
    >>> bb = [
    ...       ({ "l": jnp.array([1, 2]), "r": jnp.array(5) },
    ...        jnp.array(7)),
    ...       ({ "l": jnp.array([3, 4]), "r": jnp.array(6) },
    ...        jnp.array(8))
    ...      ]
    >>> chex.assert_trees_all_equal(aa, bb)
    """
    treedef: PyTreeDef  # to make mypy happy
    leaves, treedef = jt.flatten(tree)
    n_trees = leaves[0].shape[0]
    for i in range(n_trees):
        new_leaves = [leaf[i] for leaf in leaves]
        yield treedef.unflatten(new_leaves)
