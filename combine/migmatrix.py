import numpy as np
import matplotlib.pyplot as plt

#all 4
# totals = [10.039, 18.008, 7.115, 10.548]
#ggf only
totals = [7.633, 14.681, 5.725]

# x = np.random.randn(1000) 
# y = np.random.randn(1000) 

x_edges = np.array([0, 1, 2, 3])  # 4 bins on the x-axis
y_edges = np.array([0, 1, 2, 3])  # 4 bins on the y-axis
# x_edges = np.array([0, 1, 2, 3, 4])  # 4 bins on the x-axis
# y_edges = np.array([0, 1, 2, 3, 4])  # 4 bins on the y-axis

# Define the explicit counts for each bin (4x4 matrix)
# data = np.array([
#     [4.355/totals[0], 3.278/totals[0], 0.0, 2.406/totals[0]],
#     [0.850/totals[1], 12.346/totals[1], 1.485/totals[1], 3.327/totals[1]],
#     [0.0, 0.476/totals[2], 5.249/totals[2], 1.390/totals[2]],
#     [0.261/totals[3], 1.431/totals[3], 0.530/totals[3], 8.326/totals[3]]
# ])
#ggf only
data = np.array([
    [4.355/totals[0], 3.278/totals[0], 0.0],
    [0.850/totals[1], 12.346/totals[1], 1.485/totals[1]],
    [0.0, 0.476/totals[2], 5.249/totals[2]]
])

# Create the x and y data points to replicate the desired bin contents
fig, ax = plt.subplots(figsize=(8, 6))
mesh = ax.pcolormesh(x_edges, y_edges, data, cmap='Blues', shading='auto')

xlabels= ['ggF gen pT 200-300', 'ggF gen pT 300-450', 'ggF gen pT 450-Inf']
ylabels= ['ggF reco pT 250-350', 'ggF reco pT 350-500', 'ggF reco pT 500-Inf']
# xlabels= ['ggF gen pT 200-300', 'ggF gen pT 300-450', 'ggF gen pT 450-Inf', 'VBF']
# ylabels= ['ggF reco pT 250-350', 'ggF reco pT 350-500', 'ggF reco pT 500-Inf', 'VBF']

# Create a 2D histogram with 9 equally spaced bins
# fig, ax = plt.subplots(figsize=(8, 6))
# # counts, x_edges, y_edges, im = ax.hist2d(x, y, bins=(4, 4), range=[[0, 4], [0, 4]], cmap='Blues')
# hist, x_edges, y_edges, im = ax.hist2d(x, y, bins=[x_edges, y_edges], cmap='Blues')

# Add a colorbar to show the intensity scale
fig.colorbar(mesh, ax=ax, label='Rate')

# Customize the x-axis and y-axis with text labels
x_tick_labels = [xlabels[i] for i in range(len(x_edges) - 1)]
y_tick_labels = [ylabels[i] for i in range(len(y_edges) - 1)]

#label bins
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        x_center = (x_edges[j] + x_edges[j + 1]) / 2
        y_center = (y_edges[i] + y_edges[i + 1]) / 2
        ax.text(x_center, y_center, str(round(data[i, j],2)),
                ha='center', va='center', color='black', fontsize=10)

# Set the ticks at the center of each bin and assign the custom labels
ax.set_xticks((x_edges[:-1] + x_edges[1:]) / 2)
ax.set_yticks((y_edges[:-1] + y_edges[1:]) / 2)

ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(y_tick_labels, fontsize=10)

# Label the axes
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
plt.title('Migration Matrix')

# Display the plot
plt.show()
