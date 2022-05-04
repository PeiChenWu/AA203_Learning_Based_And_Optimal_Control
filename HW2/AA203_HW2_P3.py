import numpy as np, tensorflow as tf, matplotlib.pyplot as plt

n = 20
m = n
sdim, adim = m * n, 4

# the parameters of the storm
x_eye, sig = np.array([15, 15]), 10

x_goal = (19,19)

w_fn = lambda x: np.exp(-((np.linalg.norm(np.array(x) - x_eye)**2)/(2*sig**2)))


def helpful_matrices():
    y, x = np.meshgrid(np.arange(n), np.arange(m))
    idx2pos = np.stack([x.reshape(-1), y.reshape(-1)], -1)
    pos2idx = np.reshape(np.arange(m * n), (m, n))
    return idx2pos, pos2idx


def make_transition_matrices():
    
    pt_min, pt_max = [0, 0], [m - 1, n - 1]
    
    idx2pos, pos2idx = helpful_matrices()

    T_right, T_up, T_left, T_down = [np.zeros((sdim, sdim)) for _ in range(4)]
    
    for i in range(m):
        for j in range(n):
            w = w_fn(np.array([i,j])) # w at current pt
            
            right = np.clip(np.array([i,j]) + np.array([1, 0]), pt_min, pt_max)
            up = np.clip(np.array([i,j]) + np.array([0, 1]), pt_min, pt_max)
            left = np.clip(np.array([i,j]) + np.array([-1, 0]), pt_min, pt_max)
            down = np.clip(np.array([i,j]) + np.array([0, -1]), pt_min, pt_max)

            # specify action = right
            T_right[pos2idx[i, j], pos2idx[right[0], right[1]]] = 1 - w
            T_right[pos2idx[i, j], pos2idx[up[0], up[1]]] = w / 3
            T_right[pos2idx[i, j], pos2idx[left[0], left[1]]] = w / 3
            T_right[pos2idx[i, j], pos2idx[down[0], down[1]]] = w / 3

            # specify action = up
            T_up[pos2idx[i, j], pos2idx[right[0], right[1]]] = w / 3
            T_up[pos2idx[i, j], pos2idx[up[0], up[1]]] = 1 - w
            T_up[pos2idx[i, j], pos2idx[left[0], left[1]]] = w / 3
            T_up[pos2idx[i, j], pos2idx[down[0], down[1]]] = w / 3

            # specify action = left
            T_left[pos2idx[i, j], pos2idx[right[0], right[1]]] = w / 3
            T_left[pos2idx[i, j], pos2idx[up[0], up[1]]] = w / 3
            T_left[pos2idx[i, j], pos2idx[left[0], left[1]]] = 1 - w
            T_left[pos2idx[i, j], pos2idx[down[0], down[1]]] = w / 3

            # specify action = down
            T_down[pos2idx[i, j], pos2idx[right[0], right[1]]] = w / 3
            T_down[pos2idx[i, j], pos2idx[up[0], up[1]]] = w / 3
            T_down[pos2idx[i, j], pos2idx[left[0], left[1]]] = w / 3
            T_down[pos2idx[i, j], pos2idx[down[0], down[1]]] = 1 - w
            
    return pos2idx, idx2pos, (T_right, T_up, T_left, T_down)


def value_iteration(Ts, reward, mask, gam):

    V = tf.zeros([sdim])
    
    # perform value iteration
    for _ in range(1000):
        V_stack = tf.stack([V,V,V,V],-1)
        V_stack = tf.transpose(V_stack, perm=[1,0])
        V_prev = V
        
        V = tf.math.reduce_max(reward + tf.reshape(gam*(Ts@V_stack[:,:,None])*mask[None,:,None],[4,400]), axis=0)
        
        err = tf.linalg.norm(V-V_prev)
        
        if err < 1e-8:
            break

    return V



def visualize_value_function(V):
    pt_min, pt_max = [0, 0], [m - 1, n - 1]
    V = np.array(V)
    
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    
    u, v = [], []
    for pt in pts:
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_up, pt_down, pt_left, pt_right]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])
    u, v = np.reshape(u, (m, n)), np.reshape(v, (m, n))

    plt.imshow(V.T, origin="lower")
    plt.quiver(X, Y, u, v, pivot="middle")



def visualize_value_function_color(V):
    pt_min, pt_max = [0, 0], [m - 1, n - 1]
    V = np.array(V)

    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    
    u, v, opt_act_list = [], [], []
    for pt in pts:
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
    pt_min, pt_max = [0, 0], [m - 1, n - 1]
    V = np.array(V)
    
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    
    u, v = [], []
    for pt in pts:
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
    
    
    u, v = [], []
    optimal_policy = [pts[380]] # start point (0,19)
    pt = pts[380]
    for _ in range(100):
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        
        rd = np.random.uniform(0,1) # with w probabality, the storm will cause the drone to move in a uniformly random direction.
        if rd < w_fn(pt):
            idx = int(np.random.choice(np.delete(np.array([0,1,2,3]), idx), size=1))
            
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])

        optimal_policy.append(np.array([next_pts[idx][0],next_pts[idx][1]]))
        pt = next_pts[idx]

    plt.quiver(np.array(optimal_policy)[:-1,0],np.array(optimal_policy)[:-1,1], u, v, pivot="middle", color='red')


pos2idx, idx2pos, Ts = make_transition_matrices()

Ts = [tf.convert_to_tensor(T, dtype=tf.float32) for T in Ts]
Ts = tf.convert_to_tensor(Ts,dtype=tf.float32)

# create the terminal mask vector
mask = np.zeros([sdim])
mask[pos2idx[19, 9]] = 1.0
mask = tf.convert_to_tensor(mask, dtype=tf.float32)
mask = tf.cast(mask==0, dtype=tf.float32)

# generate the reward vector
reward = np.zeros([sdim, 4])
reward[pos2idx[19, 9], :] = 1.0
reward = tf.convert_to_tensor(reward, dtype=tf.float32)
reward = tf.transpose(reward, perm=[1,0])


gam = 0.95
V_opt = value_iteration(Ts, reward, mask, gam)

plt.figure(figsize=(10, 10))
visualize_value_function(np.array(V_opt).reshape((n, n)))
plt.title("value iteration")
plt.show()

plt.figure(figsize=(10, 10))
visualize_value_function_color(np.array(V_opt).reshape((n, n)))
plt.title("value iteration (color)")
plt.show()

plt.figure(figsize=(10, 10))
visualize_value_function_with_Stochasticity(np.array(V_opt).reshape((n, n)))
plt.title("value iteration w. Stochasticity")
plt.show()