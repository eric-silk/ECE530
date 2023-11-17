import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

label_conversion_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

np.set_printoptions(precision=3)


def get_G1() -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from((1, 2, 3, 4, 5))
    g.add_edges_from(((1, 2), (2, 4), (4, 3), (3, 5), (3, 1)))

    return g


def get_G2() -> nx.Graph:
    g = get_G1()
    g.add_edge(1, 4)

    return g


def plot_graph(G, label_conversion=label_conversion_dict):
    pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, label_conversion)


def one_a():
    # The definition of the Laplacian given by the HW assigns a weight to the
    # node as well, so we must add some positive diagonal matrix (use I for
    # convenience)
    laplacian = nx.laplacian_matrix(get_G1()).toarray() + np.eye(5)
    print(f"1a Laplacian:\n{laplacian}")
    eigvals = np.linalg.eigvals(laplacian)
    print(f"1a eigenvalues:\n{eigvals}")
    print(f"1a PD: {not np.any(np.isclose(eigvals, 0.0)) and np.all(eigvals > 0)}")
    plot_graph(get_G1(), label_conversion=None)
    plt.savefig("one_a.pdf")


def one_b():
    laplacian = nx.laplacian_matrix(get_G1()).toarray() + np.eye(5)
    L = np.linalg.cholesky(laplacian)
    print("1b L:")
    print(L)
    G_of_L = nx.from_numpy_array(L, create_using=nx.Graph)
    G_of_L.remove_edges_from(nx.selfloop_edges(G_of_L))
    plt.figure()
    plot_graph(G_of_L)
    plt.savefig("one_b.pdf")


def one_c():
    P = np.array(
        (
            (0, 0, 0, 0, 1),
            (0, 1, 0, 0, 0),
            (0, 0, 1, 0, 0),
            (1, 0, 0, 0, 0),
            (0, 0, 0, 1, 0),
        )
    )
    laplacian = nx.laplacian_matrix(get_G1()).toarray() + np.eye(5)
    reordered = P @ laplacian @ P.T
    L = np.linalg.cholesky(reordered)
    G_of_L = nx.from_numpy_array(L, create_using=nx.Graph)
    G_of_L.remove_edges_from(nx.selfloop_edges(G_of_L))
    plt.figure()
    plot_graph(G_of_L, label_conversion=label_conversion_dict)
    plt.savefig("one_c.pdf")


if __name__ == "__main__":
    one_a()
    one_b()
    one_c()
