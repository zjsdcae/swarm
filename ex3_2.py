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

def get_rigidity_matrix(E, pos):
    # E for the origin graph, pos for the position and N for the number of nodes
    len_row = len(E)
    len_col = len(pos)
    N = int(len_col/2)
    pos_vec = np.reshape(pos,(N,2))
    
    rigidity_matrix = np.matrix(np.zeros((len_row,len_col)))
    pos_re = np.reshape(pos,(N,2))
    
    for i,edge in enumerate(E):
        m1, m2 = edge[0], edge[1]
        [rigidity_matrix[i, 2*m1], rigidity_matrix[i,2*m1+1]] = 2*(pos_re[m1]-pos_re[m2])
        [rigidity_matrix[i, 2*m2], rigidity_matrix[i,2*m2+1]] = -2*(pos_re[m1]-pos_re[m2])
    return rigidity_matrix

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

def gd(E, pos):
    constrains = []
    pos_each = np.reshape(pos,(int(len(pos)/2),2))
    for edge in E:
        constrains.append(np.sum(np.square(np.subtract(pos_each[edge[0]],pos_each[edge[1]]))))
    return constrains

def make_animation(plotx,E,xl=(-2,2),yl=(-2,2),inter=20, display=False):
    '''
    takes a graph and motion of vertexes in 2D and returns an animation
    E: list of edges (each edge is a pair of vertexes)
    plotx: a matrix of states ordered as (x1, y1, x2, y2, ..., xn, yn) in the rows and time in columns
    xl and yl define the display boundaries of the graph
    inter is the interval between each point in ms
    '''
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

#    ani.save("test.mv4")
    
    plt.show(fig)
#    plt.close(fig)
#    plt.close(ani._fig)
    plt.close(all)

    if(display==True):
        IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))
    return ani

def formation_ctrl(pos, gd_des, E, T, dt=0.001):
    time_arr = np.arange(0,T,dt)

    p = np.zeros((len(pos),len(time_arr)))
    p[:,0] = pos

    for i in range(len(time_arr)-1):
        current_constraints = gd(E,p[:,i])
        current_rigidity = np.transpose(get_rigidity_matrix(E,p[:,i]))
        p[:,i+1] = np.matmul(current_rigidity,np.subtract(gd_des,current_constraints))*dt + p[:,i]
    return p

#initial the random position for the four nodes
def pos_init(N): return (np.random.random(2*N)-0.5)*2

if __name__ == "__main__":
    E = [[0,1],[0,2],[0,3],[1,2],[2,3]]
    T = 10
    dt = 0.001
    pos_des = [0.5,1.5,1.5,1.5,1.5,0.5,0.26,0.26]
    gd_des = gd(E,pos_des)
    position_init = pos_init(4)
    
    print(position_init)

    res = formation_ctrl(position_init, gd_des, E, T, dt=dt)
    plotx = res[:,::20]
    make_animation(plotx,E,inter=50, display=True)
