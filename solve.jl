include("FlowSolver.jl")

using .FlowSolver

Nz = 21 
H = 4000.0
dz = H / Nz

f = 1e-4

z_W = collect(range(0, - H, length=Nz+1))
z_T = (z_W[1:end-1] + z_W[2:end]) / 2.0

dz = z_W[1] - z_W[2]
dz_W = [ dz for i=1:Nz+1 ]
dz_T = [ dz for i=1:Nz   ]

k = π/H

f2 = 1.0
K2 = 1/k^4/30

ψ(z) = sin(k*z) + sin(3*k*z) + cos(2*k*z) + exp(z*k) * 1
d2ψdz2(z) = - k^2 * sin(k*z) - (3*k)^2 * sin(3*k*z) - (2*k)^2 * cos(2*k*z) + k^2 * exp(z*k) * 1 
d4ψdz4(z) = + k^4 * sin(k*z) + (3*k)^4 * sin(3*k*z) + (2*k)^4 * cos(2*k*z) + k^4 * exp(z*k) * 1
d6ψdz6(z) = - k^6 * sin(k*z) - (3*k)^6 * sin(3*k*z) - (2*k)^6 * cos(2*k*z) + k^6 * exp(z*k) * 1

#ψ(z) = exp(z*k) * 1
#d2ψdz2(z) = k^2 * exp(z*k) * 1 
#d4ψdz4(z) = k^4 * exp(z*k) * 1
#d6ψdz6(z) = k^6 * exp(z*k) * 1


#ψ(z) = sin(k*z) + sin(3*k*z)
#d2ψdz2(z) = - k^2 * sin(k*z) - (3*k)^2 * sin(3*k*z) #- (2*k)^2 * cos(2*k*z) + k^2 * exp(z*k) * 5 
#d4ψdz4(z) = + k^4 * sin(k*z) + (3*k)^4 * sin(3*k*z) #+ (2*k)^4 * cos(2*k*z) + k^4 * exp(z*k) * 5
#d6ψdz6(z) = - k^6 * sin(k*z) - (3*k)^6 * sin(3*k*z) #- (2*k)^6 * cos(2*k*z) + k^6 * exp(z*k) * 5

RHS(z) = f2 * d2ψdz2(z) + K2 * d6ψdz6(z)

mops = makeOperators(dz_W, dz_T)
rhs = RHS.(z_W)

ψ_num = solve(
    rhs,
    mops;
    ψ_t = ψ(z_W[1]),
    ψ_b = ψ(z_W[end]),
    d2ψ_t = d2ψdz2(z_W[1]),
    d2ψ_b = d2ψdz2(z_W[end]),
    d4ψ_t = d4ψdz4(z_W[1]),
    d4ψ_b = d4ψdz4(z_W[end]),
    f2 = f2,
    K2 = K2,
)

ψ_true = ψ.(z_W)

println("Importing PyPlot...")
using PyPlot
plt = PyPlot
println("Done")

fig, ax = plt.subplots(1, 4, sharey=true)

ax[1].plot(ψ_true, z_W, "k-", label="Ans")
ax[2].plot(ψ_num, z_W, "r" , label="Num")
ax[3].plot(ψ_num - ψ_true, z_W, "b-" , label="Num - True")
ax[4].plot(rhs, z_W, label="RHS")

for _ax in ax
    _ax.legend()
end

ax[1].set_title("\$\\Psi\$ true")
ax[2].set_title("\$\\Psi\$ solved")
ax[3].set_title("Error")
ax[4].set_title("RHS")
plt.show()





