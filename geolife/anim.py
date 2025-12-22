
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from .glds import NTrajectorySet, xl, xh, yl, yh, sample_interval 

def view_trajectory(
        basedir,
        keys,                       # name of csv file
        frame_interval=5,           # animation interval
        move_trail=15,              # trailing movement          
        zoom=False,     
    ):



    ds = NTrajectorySet(basedir=basedir)
    nds=len(ds)
    nkeys = keys.split(',') if keys else ds.keys
    for itraj in nkeys:
        if not itraj: continue

        traj,key=ds(itraj)
        x,y = traj[:,0], traj[:,1]
        print(key, len(traj))

        fig, ax = plt.subplots()
        traj_duration = sample_interval*len(traj)/3600
        ax.set_title(f'#{itraj}/{key}.csv/{len(traj)} points/{traj_duration:.3f} hours')
        ax.plot(x, y,  marker='s', color='black')
        ax.plot(x[0], y[0],  marker='o', color='tab:red')
        ax.plot(x[-1], y[-1],  marker='o', color='tab:green')
        xdata, ydata = [], []
        ln, = ax.plot([], [], marker='.', color='gold')


        def init():
            if not zoom:
                ax.set_xlim(xl, xh)
                ax.set_ylim(yl, yh)
            ax.grid(axis='both')
            return ln,

        def update(frame):
        
            xdata.append(x[frame])
            ydata.append(y[frame])


            ln.set_data(xdata, ydata)

            if len(xdata)>move_trail:  xdata.pop(0), ydata.pop(0)
            return ln,

        ani = FuncAnimation(fig, update, frames=np.arange(len(x)), interval=frame_interval, init_func=init, blit=True)
        #ani.save(f'f{iuser}.{iday}.gif', writer='imagemagick', fps=30)
        plt.show()
        #input()