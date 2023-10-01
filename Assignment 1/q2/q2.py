#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd

np.random.seed(42)

# Given parameters
n_samples = 1000000  
theta_0 = 3  
theta_1 = 1  
theta_2 = 2  
noise_variance = 2  

x1 = np.random.normal(3, 2, n_samples)  # x1 ~ N(3, 4)
x2 = np.random.normal(-1, 2, n_samples)  # x2 ~ N(-1, 4)
noise = np.random.normal(0, np.sqrt(noise_variance), n_samples)  # noise ~ N(0, sqrt(2))

y = theta_0 + theta_1 * x1 + theta_2 * x2 + noise

df = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'y': y
})

df.describe(), df.head()


# In[43]:


def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def stochastic_gradient_descent(X, y, lr=0.001, batch_sizes=[1], epochs=10000, tol=1e-9):
    m, n = X.shape  # Number of samples and features
    results = {}  # Dictionary to store results for different batch sizes
    
    for batch_size in batch_sizes:
        theta = np.zeros(n)
        loss_history = [] 
        theta_history = []
        num_batches = m // batch_size
        
        
        
        print(f"Running SGD for batch size = {batch_size}...")
        
        for epoch in range(epochs):
            # Shuffle data initially
            shuffle_idx = np.random.permutation(m)
            X_shuffled, y_shuffled = X[shuffle_idx], y[shuffle_idx]
            for b in range(num_batches):
                theta_history.append(theta.copy())
                
                start_idx = b * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                y_pred = X_batch.dot(theta)
                gradient = -2 * X_batch.T.dot(y_batch - y_pred) / batch_size
                
                theta -= lr * gradient
                
            y_pred_epoch = X.dot(theta)
            loss = compute_loss(y, y_pred_epoch)
            loss_history.append(loss)
            
            if epoch > 1 and abs(loss_history[-1] - loss_history[-2]) < tol and abs(loss_history[-2] - loss_history[-3]) < tol:
                print(f"Converged at epoch {epoch} for batch size = {batch_size}")
                break
        
        results[batch_size] = {
            'theta': theta,
            'loss_history': loss_history,
            'theta_history': theta_history,
            'epochs': epoch + 1  # +1 because epoch is 0-indexed
        }
        
    return results

X = np.c_[np.ones(n_samples), df[['x1', 'x2']].values]
y = df['y'].values

batch_sizes_to_try = [1, 100, 10000, 1000000]

sgd_results = stochastic_gradient_descent(X, y, batch_sizes=batch_sizes_to_try)

learned_thetas = {batch_size: res['theta'] for batch_size, res in sgd_results.items()}
learned_thetas


# In[44]:


test_df = pd.read_csv('../data/q2/q2test.csv')

X_test = np.c_[np.ones(test_df.shape[0]), test_df[['X_1', 'X_2']].values]
y_test = test_df['Y'].values

def compute_test_error(X, y, theta):
    y_pred = X.dot(theta)
    return compute_loss(y, y_pred)

test_errors = {}
for batch_size, result in sgd_results.items():
    theta = result['theta']
    test_error = compute_test_error(X_test, y_test, theta)
    test_errors[batch_size] = test_error

original_theta = np.array([theta_0, theta_1, theta_2])
original_test_error = compute_test_error(X_test, y_test, original_theta)

test_errors, original_test_error


# In[45]:


result['theta_history']


# In[46]:


result['theta_history'] = np.array(result['theta_history'])


# In[47]:


result['theta_history'].shape


# In[48]:


sgd_results[1]['theta_history']


# In[53]:


sgd_results[1]['theta_history'][::100000]


# In[82]:


print(len(np.array(sgd_results[1]['theta_history'][::1000])))
print(len(np.array(sgd_results[100]['theta_history'][::1000])))
print(len(np.array(sgd_results[10000]['theta_history'][::1000])))
print(len(np.array(sgd_results[1000000]['theta_history'][::1000])))


# In[78]:


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# # Function to visualize 3D movement of a single theta_history
# def visualize_single_theta_3D(theta_history):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
# 
#     # Set labels
#     ax.set_xlabel('Theta 0')
#     ax.set_ylabel('Theta 1')
#     ax.set_zlabel('Theta 2')
# 
#     line, = ax.plot([], [], [])
#     point, = ax.plot([], [], [], 'o')
# 
#     # Initialize line and point
#     def init():
#         line.set_data([], [])
#         line.set_3d_properties([])
#         point.set_data([], [])
#         point.set_3d_properties([])
#         return line, point
# 
#     # Update function for animation
#     def update(frame):
#         line.set_data(theta_history[:frame+1, 0], theta_history[:frame+1, 1])
#         line.set_3d_properties(theta_history[:frame+1, 2])
#         point.set_data(theta_history[frame, 0], theta_history[frame, 1])
#         point.set_3d_properties(theta_history[frame, 2])
#         return line, point
#     print(len(theta_history))
#     ani = animation.FuncAnimation(fig, update, frames=len(theta_history), init_func=init, interval=0.2*1000)
#     ani.save('single_3D_trajectory_line.gif', writer=PillowWriter(fps=5))
#     plt.show()
# 
# # Replace with your actual theta_history
# # For demonstration, I'm using random data
# dummy_theta_history = np.random.randn(50, 3)  # Replace with your theta_history
# dummy_theta_history = np.array(sgd_results[1]['theta_history'][::60000])
# print(dummy_theta_history)
# 
# # visualize_single_theta_3D(np.array(sgd_results[1]['theta_history'][::60000]))
# visualize_single_theta_3D(dummy_theta_history)

def visualize_single_theta_3D(theta_history, filename):
    print("size of theta history is", len(theta_history))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set labels
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Theta 2')

    # Set the axes limits based on the min and max values of theta_history
    ax.set_xlim(np.min(theta_history[:, 0]), np.max(theta_history[:, 0]))
    ax.set_ylim(np.min(theta_history[:, 1]), np.max(theta_history[:, 1]))
    ax.set_zlim(np.min(theta_history[:, 2]), np.max(theta_history[:, 2]))

    line, = ax.plot([], [], [])
    point, = ax.plot([], [], [], 'o')

    # Initialize line and point
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    # Update function for animation
    def update(frame):
        line.set_data(theta_history[:frame+1, 0], theta_history[:frame+1, 1])
        line.set_3d_properties(theta_history[:frame+1, 2])
        point.set_data(theta_history[frame, 0], theta_history[frame, 1])
        point.set_3d_properties(theta_history[frame, 2])
        return line, point

    ani = animation.FuncAnimation(fig, update, frames=len(theta_history), init_func=init, interval=0.2*1000)
    ani.save('3D_trajectory_line'+filename+'.gif', writer=PillowWriter(fps=5))
    plt.show()

# Replace with your actual theta_history
# For demonstration, I'm using random data
# dummy_theta_history = np.random.randn(50, 3)  # Replace with your theta_history
# 
# visualize_single_theta_3D(np.array(sgd_results[1]['theta_history'][::60000]))


# In[81]:


for batch_sz in batch_sizes_to_try[::-1]:
   visualize_single_theta_3D(np.array(sgd_results[batch_sz]['theta_history'][::1000]), 
                             "batch size_" + str(batch_sz)) 

