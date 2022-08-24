include("MatrixOperators.jl")

using SparseArrays

function make_ops(bmo, dz)

    Nz = bmo.Nz

    tmp = zeros(Float64, bmo.W_pts)
    tmp[2:end-1] .= 1.0

    W_mask_W = spdiagm(0 => tmp)
        
    T_∂z_W = (1/dz) * (bmo.T_DN_W - bmo.T_UP_W)
    W_∂z_T = (1/dz) * W_mask_W * (bmo.W_DN_T - bmo.W_UP_T)

    W_∂z2_W = W_∂z_T * T_∂z_W
    W_∂z4_W = W_∂z2_W * W_∂z2_W
    W_∂z6_W = W_∂z2_W * W_∂z4_W


    eW_send_W = W_mask_W[2:end-1, :]
    W_send_eW = transpose(eW_send_W)



    return (
        W_mask_W = W_mask_W,
        eW_send_W = eW_send_W,
        W_send_eW = W_send_eW,
        T_∂z_W = T_∂z_W,
        W_∂z_T = W_∂z_T,
        W_∂z2_W = W_∂z2_W,
        W_∂z4_W = W_∂z4_W,
        W_∂z6_W = W_∂z6_W,
    )

end


Nz = 200
H = 1000.0
dz = H / Nz
z_W = collect(range(0, - H, length=Nz+1))
z_T = (z_W[1:end-1] + z_W[2:end]) / 2.0

bmo = MatrixOperators(;Ny=1, Nx=1, Nz=Nz)
amo = make_ops(bmo, dz)

k = π/H
c = 1/k^4/30
ψ(z) = sin(k*z) + sin(3*k*z) + cos(2*k*z) + exp(z*k) * 5
d2ψdz2(z) = - k^2 * sin(k*z) - (3*k)^2 * sin(3*k*z) - (2*k)^2 * cos(2*k*z) + k^2 * exp(z*k) * 5 
d4ψdz4(z) = + k^4 * sin(k*z) + (3*k)^4 * sin(3*k*z) + (2*k)^4 * cos(2*k*z) + k^4 * exp(z*k) * 5
d6ψdz6(z) = - k^6 * sin(k*z) - (3*k)^6 * sin(3*k*z) - (2*k)^6 * cos(2*k*z) + k^6 * exp(z*k) * 5

RHS(z) = d2ψdz2(z) + c * d6ψdz6(z)

RHS1(z) = d2ψdz2(z)
RHS2(z) = c * d6ψdz6(z)


_RHS = RHS.(z_W)

_bc_4 = z_W * 0
_bc_4[1]   = d4ψdz4(z_W[1]) 
_bc_4[end] = d4ψdz4(z_W[end])

_bc_2 = z_W * 0
_bc_2[1]   = d2ψdz2(z_W[1]) 
_bc_2[end] = d2ψdz2(z_W[end]) 

op_LHS = amo.eW_send_W * (bmo.W_I_W + c * amo.W_∂z2_W * amo.W_∂z2_W) * amo.W_∂z2_W * amo.W_send_eW

ψ_num = amo.W_send_eW * (op_LHS \ ( amo.eW_send_W * ( _RHS - c * amo.W_∂z2_W * _bc_4 - c * amo.W_∂z4_W * _bc_2) ))

ψ_num .+= (ψ(z_W[1]) - ψ(z_W[end])) * z_W / H .+ ψ(z_W[1])

#ψ_num = op_LHS \ ( _RHS - amo.W_∂z2_W * _bc_0 - c * amo.W_∂z2_W * _bc_4 - c * amo.W_∂z4_W * _bc_2) 

println("Importing PyPlot...")
using PyPlot
plt = PyPlot
println("Done")

fig, ax = plt.subplots(1, 2, sharey=true)

ax[1].plot(ψ.(z_W), z_W, "k-", label="Ans")
ax[1].plot(ψ_num, z_W, "r--" , label="Num")
ax[1].plot(ψ_num - ψ.(z_W), z_W, "b-" , label="Num-Ans")
ax[2].plot(RHS.(z_W), z_W, label="RHS")
ax[2].plot(RHS1.(z_W), z_W, label="1")
ax[2].plot(RHS2.(z_W), z_W, label="2")

ax[1].legend()
ax[2].legend()

ax[1].set_title("\$\\Psi\$")
ax[2].set_title("RHS")
plt.show()





