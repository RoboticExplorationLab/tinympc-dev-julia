import Pkg;
Pkg.activate(@__DIR__);
Pkg.instantiate()

using RobotZoo:Cartpole
using RobotDynamics
using ForwardDiff
using TrajOptPlots
using BlockDiagonals
using LinearAlgebra
using StaticArrays
using SparseArrays
using ControlSystems

using MeshCat
using ColorTypes
using GeometryBasics: HyperSphere, HyperRectangle, Cylinder, Vec, Point, Mesh
using CoordinateTransformations
using Rotations

using Plots
using Printf

include("../quaternion-stuff.jl")

##

#Cartpole Dynamics
a = Cartpole()
h = 1/100

function dynamics_rk4(x,u)
    #RK4 integration with zero-order hold on u
    f1 = RobotDynamics.dynamics(a, x, u)
    f2 = RobotDynamics.dynamics(a, x + 0.5*h*f1, u)
    f3 = RobotDynamics.dynamics(a, x + 0.5*h*f2, u)
    f4 = RobotDynamics.dynamics(a, x + h*f3, u)
    return x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
end

##

Nx = 4     # number of state
Nu = 1     # number of controls
Tfinal = 5.0 # final time
Nt = Int(Tfinal/h)+1    # number of time steps
thist = Array(range(0,h*(Nt-1), step=h));

##

# Cost weights
Q = collect(Diagonal([1.0*ones(2); 1.0*ones(2)]));
R = 0.1;
Qn = Array(100.0*I(Nx));

##

# Goal state
xg = [0; pi/2; 0; 0];

##

# Linearized state and control matrices
A = ForwardDiff.jacobian(dx->dynamics_rk4(dx, 0), xg)
B = ForwardDiff.derivative(du->dynamics_rk4(xg, du), 0)
display(A)
display(B)

##

K = dlqr(A,B,Q,R)

