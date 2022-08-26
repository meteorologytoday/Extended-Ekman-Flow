include("FlowSolver.jl")

using .FlowSolver
using Formatting

Nz = 50 
H = 500.0
dz = H / Nz

Ω = 7.292e-5
lat = 10.0  # in degrees north

f = 2 * Ω * sin(deg2rad(lat))
K = 0.05 # 1e-5 * 4000 
ρ0 = 1026.0

z_W = collect(range(0, - H, length=Nz+1))
z_T = (z_W[1:end-1] + z_W[2:end]) / 2.0

dz = z_W[1] - z_W[2]
dz_W = [ dz for i=1:Nz+1 ]
dz_T = [ dz for i=1:Nz   ]

f2 = f^2
K2 = K^2
τx = - 0.01 # N/m^2
τy = 0*0.001 # N/m^2

#N*s / m^2 / ρ = kg * m/s^2 * s / m^2  / (kg / m^3) = kg * m/s / m^2 * m^3 / kg = m/s * m = m^3 / s  / m = Sv/m

Ek_transport = - τx / (ρ0 * f)
println(format("Theoretical integrated flux: τx / (ρ0 f) =  {:.3e} m^3/s / m", Ek_transport))



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
    ψ_b = 0.0,#- τx / f / ρ0,
    d2ψ_t = - τy / K / ρ0,
    d2ψ_b = 0.0,
    d4ψ_t = - f / K^2 * τx / ρ0,
    d4ψ_b = - f / K^2 * τx / ρ0,
    f2 = f2,
    K2 = K2,
)

v_num = - mops.T_∂z_W * ψ_num
dvdz  = mops.W_∂z_T * v_num
d3vdz3 = mops.W_∂z2_W * dvdz
dudz = d3vdz3 * K / f

println("Importing PyPlot...")
using PyPlot
plt = PyPlot
println("Done")

fig, ax = plt.subplots(1, 3, sharey=true)

fig.suptitle(format("Lat = {:.1f}", lat))

ax[1].plot(ψ_num, z_W, "r" , label="Num")
ax[2].plot(v_num * 100, z_T, "r" , label="Num")
ax[3].plot(K * dudz, z_W, label="K * dudz")

ax[1].plot([Ek_transport, Ek_transport], [-H, 0], "gray", ls="--")
ax[3].plot([0,0], [-H, 0], "gray", ls="--")

for _ax in ax
    _ax.legend()
end

ax[1].set_title("\$\\Psi\$ solved")
ax[2].set_title("\$v\$ solved")
ax[3].set_title("RHS")

ax[1].set_xlabel("[m^3/s /m]")
ax[2].set_xlabel("[cm / s]")
ax[3].set_xlabel("")

plt.show()

