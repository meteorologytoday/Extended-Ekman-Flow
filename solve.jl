include("FlowSolver.jl")

using .FlowSolver
using Formatting

Nz = 501 
H = 1000.0
dz = H / Nz

Ω = 7.292e-5
lat = 30.0  # in degree

f = 2 * Ω * sin(deg2rad(lat))
K = 5e-3 

z_W = collect(range(0, - H, length=Nz+1))
z_T = (z_W[1:end-1] + z_W[2:end]) / 2.0

dz = z_W[1] - z_W[2]
dz_W = [ dz for i=1:Nz+1 ]
dz_T = [ dz for i=1:Nz   ]

f2 = f^2
K2 = K^2
τx = 0.1 # N/m^2
τy = 0.0 # N/m^2


println(format("Theoretical integrated flux: τx / f =  {:.3e} m/s", τx / f))

# dψ/dz = - v
# d^2ψ/dz^2 = - dv/dz
# K d^2ψ/dz^2 = - K dv/dz = - τy
# d^2ψ/dz^2 = - τy / K

# d^4ψ/dz^4 = f / K^2 τx

RHS(z) = 0.0
mops = makeOperators(dz_W, dz_T)
rhs = RHS.(z_W)

ψ_num = solve(
    rhs,
    mops;
    ψ_t = 0.0,
    ψ_b = - τx / f,
    d2ψ_t = - τy / K,
    d2ψ_b = 0.0,
    d4ψ_t = - f / K^2 * τx,
    d4ψ_b =   f / K^2 * τx * 0,
    f2 = f2,
    K2 = K2,
)

v_num = - mops.T_∂z_W * ψ_num


println("Importing PyPlot...")
using PyPlot
plt = PyPlot
println("Done")

fig, ax = plt.subplots(1, 3, sharey=true)

ax[1].plot(ψ_num, z_W, "r" , label="Num")
ax[2].plot(v_num, z_T, "r" , label="Num")
ax[3].plot(rhs, z_W, label="RHS")

for _ax in ax
    _ax.legend()
end

ax[1].set_title("\$\\Psi\$ solved")
ax[2].set_title("\$v\$ solved")
ax[3].set_title("RHS")
plt.show()





