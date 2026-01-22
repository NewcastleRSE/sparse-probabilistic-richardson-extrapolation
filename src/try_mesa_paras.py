import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
from mesa import Agent, Model
from mesa.space import ContinuousSpace
from try_mesa import FlockingModel
import pandas as pd

dt_values = [0.1, 0.05, 0.02, 0.01]
sigma_values = [0.1, 0.05, 0.02, 0.01]

n_steps = 100
seed = 42
n_agents = 50
width, height = 5.0, 5.0

results = pd.DataFrame(index=dt_values, columns=sigma_values)

for dt in dt_values:
    for sigma in sigma_values:
        np.random.seed(seed)
        model = FlockingModel(
            n_agents=n_agents,
            width=width,
            height=height,
            dt=dt,
            tau=0.05,
            epsilon=0.05,
            interaction_radius=1.0,
            repulsion_radius=0.2,
            sigma=sigma,
            record_video=False
        )
        for _ in range(n_steps):
            model.step()
        model.finalize()
        results.loc[dt, sigma] = model.total_distance_from_origin()

print("Total distance from origin as dt and sigma vary:")
print(results)

# Optional heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(results.astype(float), annot=True, fmt=".2f", cmap="viridis")
plt.xlabel("sigma")
plt.ylabel("dt")
plt.title("Convergence of total distance from origin (dt vs sigma)")
plt.show()
