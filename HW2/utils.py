import math, pdb, os, sys

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt


def visualize_value_function(V):
    """
    Visualizes the value function given in V & computes the optimal action,
    visualized as an arrow.
    Args:
        V: (np.array) the value function reshaped into a 2D array.
    """
    V = np.array(V)
    assert V.ndim == 2
    m, n = V.shape
    pos2idx = np.arange(m * n).reshape((m, n))
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    u, v = [], []
    for pt in pts:
        pt_min, pt_max = [0, 0], [m - 1, n - 1]
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])
    u, v = np.reshape(u, (m, n)), np.reshape(v, (m, n))

    plt.imshow(V.T, origin="lower")
    plt.quiver(X, Y, u, v, pivot="middle")


def visualize_value_function_color(V):
    """
    Visualizes the value function given in V & computes the optimal action,
    visualized as an arrow., 
    Args:
        V: (np.array) the value function reshaped into a 2D array.
    """
    V = np.array(V)
    assert V.ndim == 2
    m, n = V.shape
    pos2idx = np.arange(m * n).reshape((m, n))
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    u, v, opt_act_list = [], [], []
    for pt in pts:
        pt_min, pt_max = [0, 0], [m - 1, n - 1]
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_up, pt_down, pt_left, pt_right]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        opt_act_list.append(idx)
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])
    u, v, opt_act = np.reshape(u, (m, n)), np.reshape(v, (m, n)), np.reshape(opt_act_list, (m, n))

    palette = np.array([[  0,   0,   0],   # black
                        [255,   0,   0],   # red
                        [  0, 255,   0],   # green
                        [  0,   0, 255]])   # blue

    for i in range(20):
        for j in range(20):
            text = plt.text(j, i, opt_act[i,j], ha="center", va="center", color='w')


    plt.imshow(palette[opt_act], origin="lower")



def visualize_value_function_with_Stochasticity(V):
    """
    Visualizes the value function given in V & computes the optimal action,
    visualized as an arrow.
    Args:
        V: (np.array) the value function reshaped into a 2D array.
    """
    V = np.array(V)
    assert V.ndim == 2
    m, n = V.shape
    pos2idx = np.arange(m * n).reshape((m, n))
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    u, v = [], []
    for pt in pts:
        pt_min, pt_max = [0, 0], [m - 1, n - 1]
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])

    u, v = np.reshape(u, (m, n)), np.reshape(v, (m, n))
    
    plt.imshow(V.T, origin="lower")
    plt.quiver(X, Y, u, v, pivot="middle")
    
    
    x_eye, sig = np.array([15, 15]), 10
    w_fn = lambda x: np.exp(-((np.linalg.norm(np.array(x) - x_eye)**2)/(2*sig**2)))
    
    u, v = [], []
    optimal_policy = [pts[380]]
    pt = pts[380]
    #while pt[0] != 19 or pt[1] != 9:
    for _ in range(100):
        pt_min, pt_max = [0, 0], [m - 1, n - 1]
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        #print(idx)
        
        rd = np.random.uniform(0,1) # with w probabality, the storm will cause the drone to move in a uniformly random direction.
        #print(w_fn(pt))
        if rd < w_fn(pt):
            #print(np.delete(np.array([0,1,2,3]), idx))
            idx = int(np.random.choice(np.delete(np.array([0,1,2,3]), idx), size=1))
            
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])

        optimal_policy.append(np.array([next_pts[idx][0],next_pts[idx][1]]))
        pt = next_pts[idx]

    plt.quiver(np.array(optimal_policy)[:-1,0],np.array(optimal_policy)[:-1,1], u, v, pivot="middle", color='red')

    
def make_transition_matrices(m, n, x_eye, sig):
    """
    Compute the transisiton matrices T, which maps a state probability vector to
    a next state probability vector.

        prob(S') = T @ prob(S)

    Args:
        n (int): the width and height of the grid
        x_eye (Sequence[int]): 2 element vector describing the storm location
        sig (float): standard deviation of the storm, increases storm size

    Returns:
        List[np.array]: 4 transition matrices for actions
                                                {right, up, left, down}
    """

    sdim = m * n

    # utility functions
    w_fn = lambda x: np.exp(-np.linalg.norm(np.array(x) - x_eye)**2 / sig ** 2 / 2)
    xclip = lambda x: min(max(0, x), m - 1)
    yclip = lambda y: min(max(0, y), n - 1)

    # graph building
    pos2idx = np.reshape(np.arange(m * n), (m, n))
    y, x = np.meshgrid(np.arange(n), np.arange(m))
    idx2pos = np.stack([x.reshape(-1), y.reshape(-1)], -1)

    T_right, T_up, T_left, T_down = [np.zeros((sdim, sdim)) for _ in range(4)]
    for i in range(m):
        for j in range(n):
            z = (i, j)
            w = w_fn(z)
            right = (xclip(z[0] + 1), yclip(z[1] + 0))
            up = (xclip(z[0] + 0), yclip(z[1] + 1))
            left = (xclip(z[0] - 1), yclip(z[1] + 0))
            down = (xclip(z[0] + 0), yclip(z[1] - 1))

            T_right[pos2idx[i, j], pos2idx[right[0], right[1]]] += 1 - w
            T_right[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_right[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_right[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_up[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_up[pos2idx[i, j], pos2idx[up[0], up[1]]] += 1 - w
            T_up[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_up[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_left[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_left[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_left[pos2idx[i, j], pos2idx[left[0], left[1]]] += 1 - w
            T_left[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_down[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[down[0], down[1]]] += 1 - w
    return (T_right, T_up, T_left, T_down), pos2idx, idx2pos


def generate_problem():
    """
    A function that generates the problem data for Problem 3.

    Generates transition matrices for each of the four actions.
    Generates pos2idx array which allows to convert from a (i, j) grid
        coordinates to a vectorized state (1D).
    """
    n = 20
    m = n
    sdim, adim = m * n, 4

    # the parameters of the storm
    x_eye, sig = np.array([15, 15]), 10

    Ts, pos2idx, idx2pos = make_transition_matrices(m, n, x_eye, sig)

    Ts = [tf.convert_to_tensor(T, dtype=tf.float32) for T in Ts]
    Problem = dict(Ts=Ts, n=n, m=m, pos2idx=pos2idx, idx2pos=idx2pos)
    return Problem
