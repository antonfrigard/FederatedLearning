using StatsBase
using Distributions
using LinearAlgebra

# Generate Xtilde and ytilde
# For each epoch
# 1) compute gradient
# 2) Generate straggling time, determine computation latency
# 3) Update model
# 4) Aggregate MSE
# Add upload and download times
# Run this experiment multiple times
#
#


l = 300 # Device number of raw data points
n = 24 # Number of devices
m = n*l
c = 400 # Number of encoded data points for each device
d = 500 # Number of features
μ = 0.0085 # Learning rate

N = Normal()
G = rand(N, (c,m))
X = rand(N, (m,d))
Xt = G*X

β = rand(N, (d,1)) # True model
z = rand(N, (m, 1)) # Noise
yt = Xt*β + G*z # Encoded labels
βr = rand(N, (d,1)) # Current guess at model

ϵ1 = norm(Xt*βr - yt)

Γ = 2/c * Xt' * (Xt*βr - yt) 

βr = βr - μ /m * Γ # Update model

ϵ2 = norm(Xt*βr - yt)



