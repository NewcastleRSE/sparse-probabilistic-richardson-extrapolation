import numpy as np
import matplotlib.pyplot as plt

# Minimal agent for demo (just moves randomly)
class DemoAgent:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.random.randn(2) * 0.1

    def step(self, dt):
        self.pos += self.vel * dt

# Simulation runner
def run_demo(dt, tau, n_steps=50, n_agents=5):
    # Initialize agents in random positions
    agents = [DemoAgent(np.random.rand(2)*10) for _ in range(n_agents)]
    # Store trajectories
    trajs = {i: [agent.pos.copy()] for i, agent in enumerate(agents)}
    last_sense_time = 0.0

    for step in range(n_steps):
        current_time = step * dt
        if current_time - last_sense_time >= tau:
            # In a real flocking model, neighbor update would occur here
            last_sense_time = current_time
        for i, agent in enumerate(agents):
            agent.step(dt)
            trajs[i].append(agent.pos.copy())
    return trajs

# Settings to compare
settings = [
    {"dt":0.2, "tau":0.2, "label":"large dt, tau"},
    {"dt":0.05,"tau":0.05,"label":"medium dt, tau"},
    {"dt":0.01,"tau":0.01,"label":"small dt, tau"}
]

# Plot trajectories
plt.figure(figsize=(10,6))
for s in settings:
    trajs = run_demo(dt=s["dt"], tau=s["tau"])
    for i, path in trajs.items():
        path = np.array(path)
        plt.plot(path[:,0], path[:,1], label=f"Agent {i}" if i==0 else "", alpha=0.6)
    plt.title(f"Trajectory example: {s['label']}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()
