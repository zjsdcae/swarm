# header to start
# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mp

# import pickle
import IPython
import ffmpeg

import scipy.linalg

def get_laplacian(E, n_vertices, directed):
    A = np.zeros([n_vertices,n_vertices]) # Adjacency
    D = np.zeros([n_vertices,n_vertices]) # Degree
    # undirected graph the graph is symmetric
    if directed == False:
        for edge in E:
            A[edge[0],edge[1]] = A[edge[1],edge[0]] = 1
    # for directed graph
    else:
        for edge in E:
            A[edge[1],edge[0]] = 1
    # the degree matrix equals the sum of the row
    for degree in range(n_vertices):
        for adj in A[degree]:
            D[degree][degree] += adj
    # Laplacian = Diag - Adjacency
    L = D - A
    return L

def make_animation(plotx,E,xl=(-5,5),yl=(-5,5),inter=20, display=False):
#    fig = mp.figure.Figure()
    fig = plt.figure()

#    mp.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=xl, ylim=yl)
    ax.grid()

    list_of_lines = []
    for i in E: #add as many lines as there are edges
        line, = ax.plot([], [], 'o-', lw=2)
        list_of_lines.append(line)

    def animate(i):
        for e in range(len(E)):
            vx1 = plotx[2*E[e][0],i]
            vy1 = plotx[2*E[e][0]+1,i]
            vx2 = plotx[2*E[e][1],i]
            vy2 = plotx[2*E[e][1]+1,i]
            list_of_lines[e].set_data([vx1,vx2],[vy1,vy2])
        return list_of_lines

    def init():
        return animate(0)

    ani = animation.FuncAnimation(fig, animate, np.arange(0, len(plotx[0,:])),
        interval=inter, blit=True, init_func=init)
    plt.show(fig)
#    plt.close(fig)
#    plt.close(ani._fig)
    plt.close(all)

    if(display==True):
        IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))
    return ani

def get_incidence(E, N):
    incidence_mat = np.zeros((N,len(E)))

    for edge in E:
        i = edge[1]
        j = edge[0]
        k = E.index(edge)
        incidence_mat[i][k] = 1
        incidence_mat[j][k] = -1

    return incidence_mat

def formation_ctrl(pos, K, E, N, z_ref, T, dt):
    L = get_laplacian(E, N, False)
    D = get_incidence(E, N)
    time_arr = np.arange(0,T,dt)
    len_time = len(time_arr)
    p = np.zeros((N,len_time))
    p[:,0] = pos

    for i in range(len_time-1):
        p[:,i+1] = (-K*np.matmul(L,p[:,i]) + K*np.matmul(D,z_ref))*dt + p[:,i]

    return p

#initial the random position for the four nodes
def pos_init(N): return (np.random.random(N)-0.5)*2

if __name__ == "__main__":
    E = [[0,1],[0,2],[0,3]]
#    E = [[0,1],[0,2],[0,3],[1,2],[2,3]]
    T = 100
    dt = 0.001
    K = 2
    N = 4
    x_0 = pos_init(4)
    y_0 = pos_init(4)
    x_des = [0.5,1.5,1.5,0.26]
    y_des = [1.5,1.5,0.5,0.26]
    z_ref_x = np.matmul(np.transpose(get_incidence(E, N)), np.transpose(x_des))
    z_ref_y = np.matmul(np.transpose(get_incidence(E, N)), np.transpose(y_des))
    
    res_x = formation_ctrl(x_0, K, E, N, z_ref_x, T, dt)
    res_y = formation_ctrl(y_0, K, E, N, z_ref_y, T, dt)
    res = np.zeros((2*N,int(T/dt)))
    for i in range(N):
        for j in range(int(T/dt)):
            res[2*i][j] = res_x[i][j]
            res[2*i+1][j] = res_y[i][j]

    plotx = res[:,::50]
    make_animation(plotx,E,inter=50, display=False)
