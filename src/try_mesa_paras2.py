import numpy as np
import pandas as pd
from try_mesa import FlockingModel

import numpy as np
import pandas as pd

# ---------------------------
# Convergence parameters
# ---------------------------
dt_values = [0.1, 0.05, 0.02, 0.01]
delta_values = [0.2, 0.1, 0.05, 0.01]

n_steps = 100
seed = 42

results = pd.DataFrame(index=dt_values, columns=delta_values)

# ---------------------------
# Run simulations
# ---------------------------
for dt in dt_values:
    for delta in delta_values:
        np.random.seed(seed)

        model = FlockingModel(
            n_agents=50,
            width=5.0,
            height=5.0,
            dt=dt,
            tau=0.05,               # fixed
            epsilon=0.05,           # fixed
            sigma=0.01,             # fixed
            delta=delta,            # <-- convergence parameter
            interaction_radius=1.0,
            repulsion_radius=0.2,
            record_video=False
        )

        for _ in range(n_steps):
            model.step()

        model.finalize()
        results.loc[dt, delta] = model.total_distance_from_origin()

# ---------------------------
# Display table
# ---------------------------
print("Total distance from origin as dt and delta vary:")
print(results)
