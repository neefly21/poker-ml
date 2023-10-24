import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###
    elems_to_assign = X.shape[0]
    
    for cs in range(elems_to_assign):
        centroid_distances = []
        for centroid in range(K):
            # X[cs] - centroids[centroid]
            norm = np.linalg.norm(X[cs] - centroids[centroid])
            centroid_distances.append(norm)
     ### END CODE HERE ###
        #idx[cs] = min(enumerate(centroid_distances))
        idx[cs] = np.argmin(centroid_distances)
    return idx

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    
    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    ### START CODE HERE ###
    
    for c in range(K):
        points = X[idx == c]
        centroids[c] = np.mean(points, axis=0)
        
    ### END CODE HERE ## 
    
    return centroids

def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))
    plot_progress_history = []
    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        plot_history = {
            "X": X,
            "centroids": centroids,
            "previous_centroids": previous_centroids,
            "idx": idx,
            "K": K,
            "i": i
        }

        if i == 9:
            plot_progress_history.append(plot_history)

        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx, plot_progress_history

def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_kmeans_subplot(k_means_history):
    plt.figure(figsize=(8, 6))
    fix, axs = plt.subplots(int(len(k_means_history)/2),1)
    cmap = ListedColormap(["red", "green", "blue"])
    
    print(len(k_means_history))

    for i in range(int(len(k_means_history)/2)):
        kmhi = k_means_history[i][0]
        c = cmap(kmhi['idx'])
        axs[i].scatter(kmhi['X'][:, 0], kmhi['X'][:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)
        axs[i].scatter(kmhi['centroids'][:, 0], kmhi['centroids'][:, 1], marker='x', c='k', linewidths=3)

        for j in range(kmhi['centroids'].shape[0]):
            c_i = kmhi['centroids'][j, :]
            p_c_i = kmhi["previous_centroids"][j, :]
            axs[i].plot([c_i[0], p_c_i[0]], [c_i[1], p_c_i[1]], "-k", linewidth=1)
    
    plt.title("Winnings v Session Length Grouping")
    plt.xlabel("$ PnL")
    plt.ylabel("Session Length(minutes)")
    plt.show()

def plot_data_points(X, idx):
    # Define colormap to match Figure 1 in the notebook
    cmap = ListedColormap(["red", "orange", "green", "blue"])
    c = cmap(idx)
    
    # plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Winnings v Session Length Grouping")
    plt.xlabel("$ PnL")
    plt.ylabel("Session Length(minutes)")

def plot_data_points_w_subplot(X, idx, spi):
    # Define colormap to match Figure 1 in the notebook
    cmap = ListedColormap(["red", "green", "blue"])
    c = cmap(idx)
    
    # plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    axs[spi].scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Winnings v Session Length Grouping")
    plt.xlabel("$ PnL")
    plt.ylabel("Session Length(minutes)")