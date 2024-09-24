import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt


# Define the neural network for PINN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))

        return self.layers[-1](x)


# Define the physics-informed loss functions
def physics_loss(
    model, x, y, inlet_pressure, membrane_permeability, inlet_flow_rate, D
):
    x.requires_grad = True
    y.requires_grad = True
    output = model(torch.cat([x, y], dim=1))
    u = output[:, 0:1]
    v = output[:, 1:2]
    c = output[:, 2:3]

    # Gradients for velocity and concentration
    u_grad = autograd.grad(
        u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True
    )

    v_grad = autograd.grad(
        v, [x, y], grad_outputs=torch.ones_like(v), create_graph=True
    )

    c_grad = autograd.grad(
        c, [x, y], grad_outputs=torch.ones_like(c), create_graph=True
    )

    u_x, u_y = u_grad[0], u_grad[1]
    v_x, v_y = v_grad[0], v_grad[1]
    c_x, c_y = c_grad[0], c_grad[1]

    # Second-order gradients
    u_xx = autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[
        0
    ]

    u_yy = autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[
        0
    ]

    v_xx = autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[
        0
    ]

    v_yy = autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[
        0
    ]

    c_xx = autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[
        0
    ]

    c_yy = autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y), create_graph=True)[
        0
    ]

    # Governing equations
    continuity = u_x + v_y
    momentum_x = u * u_x + v * u_y - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y - nu * (v_xx + v_yy)
    convection_diffusion = u * c_x + v * c_y - D * (c_xx + c_yy)
    osmotic_pressure = R * T * c
    net_transmembrane_pressure = inlet_pressure - osmotic_pressure
    water_flux = membrane_permeability * net_transmembrane_pressure

    # Boundary conditions
    inlet_bc = (
        u[:n_inlet] - inlet_velocity_profile(y[:n_inlet], inlet_flow_rate)
    ) ** 2 + (c[:n_inlet] - c_inlet) ** 2

    outlet_bc = (u[n_outlet:] - 0) ** 2 + (v[n_outlet:] - 0) ** 2

    top_membrane_bc = (u[top_membrane_idx] - 0) ** 2 + (
        v[top_membrane_idx] - water_flux[top_membrane_idx]
    ) ** 2

    bottom_membrane_bc = (u[bottom_membrane_idx] - 0) ** 2 + (
        v[bottom_membrane_idx] + water_flux[bottom_membrane_idx]
    ) ** 2

    # Total loss
    loss = torch.mean(
        continuity**2 + momentum_x**2 + momentum_y**2 + convection_diffusion**2
    )

    loss += (
        torch.mean(inlet_bc)
        + torch.mean(outlet_bc)
        + torch.mean(top_membrane_bc)
        + torch.mean(bottom_membrane_bc)
    )

    return loss


# Training function
def train(
    model,
    optimizer,
    x,
    y,
    inlet_pressure,
    membrane_permeability,
    inlet_flow_rate,
    D,
    epochs=1000,
):
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = physics_loss(
            model, x, y, inlet_pressure, membrane_permeability, inlet_flow_rate, D
        )

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")


# Define the inlet velocity profile based on the flow rate
def inlet_velocity_profile(y, flow_rate):
    return (
        (6 * flow_rate / (channel_height**2))
        * (y + channel_height / 2)
        * (channel_height / 2 - y)
    )


# Parameters
rho = 1000  # Density of water in kg/m^3
nu = 1e-6  # Kinematic viscosity of water in m^2/s
D = 1e-9  # Diffusion coefficient of salt in water in m^2/s
R = 8.314  # Universal gas constant in J/(mol*K)
T = 298  # Temperature in K
channel_height = 0.01  # Channel height in meters
c_inlet = 1  # Inlet salt concentration in mol/m^3

# Domain
n_points_x = 400  # Number of points along the x-axis
n_points_y = 40  # Number of points along the y-axis


# Non-uniform distribution in y-direction using tanh function
y = np.linspace(-channel_height / 2, channel_height / 2, n_points_y)
y = (np.tanh(y * 3) + 1) / 2 * channel_height - channel_height / 2


# Plot the distribution of collocation points
plt.figure(figsize=(10, 6))
plt.plot(y, np.zeros_like(y), "o")
plt.xlabel("y (meters)")
plt.title("Non-uniform Distribution of Collocation Points in y-direction")
plt.grid(True)

plt.show()


x = torch.linspace(0, channel_height * 10, n_points_x).reshape(
    -1, 1
)  # Length of the channel from 0 to 0.1 meters

y = torch.tensor(y).reshape(-1, 1).float()


# Identify indices for boundary conditions
n_inlet = n_points_y
n_outlet = n_points_y
top_membrane_idx = torch.arange(n_points_x * (n_points_y // 2), n_points_x * n_points_y)
bottom_membrane_idx = torch.arange(0, n_points_x * (n_points_y // 2))


# PINN model
layers = [2, 20, 20, 20, 3]  # Input: (x, y), Output: (u, v, c)
model = PINN(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training
inlet_pressure = 5e5  # Inlet pressure in Pa
membrane_permeability = 1e-11  # Membrane permeability in m/s/Pa
inlet_flow_rate = 1e-6  # Inlet flow rate in m^3/s


train(
    model,
    optimizer,
    x,
    y,
    inlet_pressure,
    membrane_permeability,
    inlet_flow_rate,
    D,
    epochs=5000,
)
