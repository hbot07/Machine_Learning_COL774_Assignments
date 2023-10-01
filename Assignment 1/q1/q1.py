#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np

x = np.loadtxt('../data/q1/linearX.csv', delimiter=',')
y = np.loadtxt('../data/q1/linearY.csv', delimiter=',')

print(x.shape, y.shape)


# In[85]:


import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.show()


# In[86]:


# normalise data
x = (x - np.mean(x)) / np.std(x)

#plot normalised data
plt.scatter(x, y)
plt.show()


# In[87]:


theta = [0 for i in range(2)]

def hypothesis(theta, x):
    return theta[0] + theta[1]*x

def cost_function(theta, x, y):
    m = len(x)
    cost = 0
    for i in range(m):
        cost += (hypothesis(theta, x[i]) - y[i])**2
    return cost/(2*m)

def gradient_descent(theta, x, y, alpha, convergence):

    m = len(x)
    cost_history = [cost_function([0,0], x, y)]
    theta_history = [[0,0]]

    converged = False

    while not converged:
        cost_history.append(cost_function(theta, x, y))
        theta_history.append(theta.copy())
        temp0 = theta[0] - (alpha/m)*sum([(hypothesis(theta, x[j]) - y[j]) for j in range(m)])
        temp1 = theta[1] - (alpha/m)*sum([(hypothesis(theta, x[j]) - y[j])*x[j] for j in range(m)])
        theta[0] = temp0
        theta[1] = temp1

        if abs(cost_function(theta, x, y) - cost_history[-1]) < convergence:
            converged = True

    return theta, cost_history, theta_history


# In[88]:


theta, cost_history, theta_history = gradient_descent(theta, x, y, 0.01, 0.1**9)

print(theta, cost_history[-1], len(cost_history))


# In[89]:


# plot line after training
plt.scatter(x, y)
plt.plot(x, hypothesis(theta, x), color='red')
plt.xlabel('Acidity (X)(Normalised)')
plt.ylabel('Density (Y)')
plt.savefig('../data/q1/plot.jpg')
plt.show()
# save plot as jpeg


# In[90]:


cost_history


# In[91]:


theta_history


# In[92]:


import matplotlib.animation as animation
import matplotlib.pyplot as plt

def hypothesis(theta, x):
    return theta[0] + theta[1] * x

# Function to draw 3D mesh and save as GIF with a moving line for trajectory
def draw_3d_mesh_and_gif_with_line(theta_history, cost_history, x, y, delay=0.2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Cost Function (J)')
    
    # Convert theta_history to numpy array for easier slicing
    theta_history = np.array(theta_history)
    
    # Create theta0 and theta1 values for the meshgrid
    theta0_vals = np.linspace(min(theta_history[:, 0]) - 1, max(theta_history[:, 0]) + 1, 100)
    theta1_vals = np.linspace(min(theta_history[:, 1]) - 1, max(theta_history[:, 1]) + 1, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
    
    # Compute the cost function for each theta value in the meshgrid
    cost_vals = np.array([cost_function([t0, t1], x, y) for t0, t1 in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
    cost_mesh = cost_vals.reshape(theta0_mesh.shape)
    
    # Plot the surface
    ax.plot_surface(theta0_mesh, theta1_mesh, cost_mesh, alpha=0.5)
    
    # Initialize line for trajectory
    line, = ax.plot([], [], [], 'r-')
    
    # Define update function for animation
    def update(num):
        line.set_data(theta_history[:num+1, 0], theta_history[:num+1, 1])
        line.set_3d_properties(cost_history[:num+1])
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=range(len(theta_history)), interval=delay*1000)
    
    # Save the animation as GIF
    ani.save('3D_trajectory_line_mesh.gif', writer=PillowWriter(fps=1 / delay))
    
    plt.show()

# Create the 3D mesh plot and GIF with a moving line for trajectory
draw_3d_mesh_and_gif_with_line(theta_history, cost_history, x, y)


# In[98]:


def draw_contour_and_gif_with_point(theta_history, cost_history, x, y, delay=0.2, filename='2D_trajectory_point_contour.gif'):
    fig, ax = plt.subplots()
    
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    
    theta_history = np.array(theta_history)
    
    theta0_vals = np.linspace(min(theta_history[:, 0]) - 1, max(theta_history[:, 0]) + 1, 100)
    theta1_vals = np.linspace(min(theta_history[:, 1]) - 1, max(theta_history[:, 1]) + 1, 100)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)
    
    cost_vals = np.array([cost_function([t0, t1], x, y) for t0, t1 in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
    cost_mesh = cost_vals.reshape(theta0_mesh.shape)
    
    CS = ax.contour(theta0_mesh, theta1_mesh, cost_mesh, levels=50, cmap='jet')
    ax.clabel(CS, inline=1, fontsize=10)
    
    point, = ax.plot([], [], 'ro')
    
    def update(num):
        point.set_data(theta_history[num, 0], theta_history[num, 1])
    
    ani = animation.FuncAnimation(fig, update, frames=range(len(theta_history)), interval=delay*1000)
    
    ani.save(filename, writer=PillowWriter(fps=1 / delay))
    
    plt.show()

# Create the 2D contour plot and GIF with a moving point for trajectory



# In[100]:


draw_contour_and_gif_with_point(theta_history, cost_history, x, y)


# In[99]:


step_sizes = [0.001, 0.025, 0.1]

for step_size in step_sizes:
    _, cost_history, theta_history = gradient_descent([0,0], x, y, step_size, 0.1**7)
    
    draw_contour_and_gif_with_point(theta_history, cost_history, x, y, 0.2, "(e) 2D Contour "+ str(step_size)+".gif")
    

