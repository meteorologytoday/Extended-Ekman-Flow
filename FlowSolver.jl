include("MatrixOperatorModule.jl")

module FlowSolver

    export makeOperators, solve
    
    using SparseArrays
    using ..MatrixOperatorModule

    function makeOperators(
        dz_W :: AbstractArray{Float64, 1},
        dz_T :: AbstractArray{Float64, 1},
    )


        Nz = length(dz_T)

        if length(dz_W) != Nz+1
            throw(ErrorException("Wrong dimension"))
        end
        
        bmo   = MatrixOperators(;Ny=1, Nx=1, Nz=Nz  )
        bmo_x = MatrixOperators(;Ny=1, Nx=1, Nz=Nz+2)   # x = extension grid

        # Be aware that idx is reusable
        extrapolate_scheme = "0th" # "0th"

        idx = zeros(Int64, bmo_x.W_pts)
        idx[2:end-1] .= collect(1:bmo.W_pts)
        idx[1] = 1
        idx[end] = bmo.W_pts
        
        Wx_extrapolate_W = bmo.W_I_W[idx, :]
       
        println("extrapolate_scheme = $extrapolate_scheme") 
        if extrapolate_scheme == "0th"

            Wx_extrapolate_W[1,       1] = 1
            Wx_extrapolate_W[end, end-1] = 1

        elseif extrapolate_scheme == "1th"

            # (x - (y-x)) x y  => 2x - y
            Wx_extrapolate_W[1,       1] =  2
            Wx_extrapolate_W[1,       2] = -1
            
            # x y (y + (y-x)) => 2y - x 
            Wx_extrapolate_W[bmo_x.W_pts, bmo.W_pts]   =  2
            Wx_extrapolate_W[bmo_x.W_pts, bmo.W_pts-1] = -1

        else
            throw(ErrorException("Unknown extrapolate_scheme : `$extrapolate_scheme`"))
        end

        idx = collect(2:bmo_x.W_pts-1)
        W_reduce_Wx = bmo_x.W_I_W[idx, :]

        #println("extrapolate")
        #println(Matrix(Wx_extrapolate_W))

        #println("reduce ")
        #println(Matrix(W_reduce_Wx))

        dz_Tx = zeros(Float64, bmo_x.T_pts)
        dz_Tx[2:end-1] = dz_T
        dz_Tx[1]   = dz_T[1]
        dz_Tx[end] = dz_T[end]
 
        dz_Wx = zeros(Float64, bmo_x.W_pts)
        dz_Wx[2:end-1] = dz_W
        dz_Wx[1]   = dz_W[1]
        dz_Wx[end] = dz_W[end]
 
        d = ones(Float64, bmo.W_pts)
        d[1] = 0.0
        d[end] = 0.0
        W_rmbnd_W = spdiagm(0 => d)
        
        Tx_∂z_Wx = spdiagm( 0 => 1 ./ dz_Tx ) * (bmo_x.T_DN_W - bmo_x.T_UP_W)
        W_∂z_Tx  = W_reduce_Wx * spdiagm( 0 => 1 ./ dz_Wx )  * (bmo_x.W_DN_T - bmo_x.W_UP_T)

        T_∂z_W = spdiagm( 0 => 1 ./ dz_T ) * (bmo.T_DN_W - bmo.T_UP_W)
        W_∂z_T = spdiagm( 0 => 1 ./ dz_W ) * W_rmbnd_W * (bmo.W_DN_T - bmo.W_UP_T)

        #W_∂z2_W = W_∂z_T * T_∂z_W
        
        W_∂z2_W = W_∂z_Tx * Tx_∂z_Wx * Wx_extrapolate_W
        W_∂z4_W = W_∂z2_W * W_∂z2_W
        W_∂z6_W = W_∂z2_W * W_∂z4_W

       
        M = W_rmbnd_W
        Δ = W_∂z2_W
        MΔ = M * Δ


        return (
            T_∂z_W = T_∂z_W,
            W_∂z_T = W_∂z_T,
            Tx_∂z_Wx = Tx_∂z_Wx,
            W_∂z_Tx  = W_∂z_Tx,
            W_∂z2_W = W_∂z2_W,
            W_∂z4_W = W_∂z4_W,
            W_∂z6_W = W_∂z6_W,
            W_rmbnd_W = W_rmbnd_W,
            M = M,
            Δ = Δ,
            MΔ = MΔ, 
        )

    end

    function solve(
        RHS :: AbstractArray{Float64, 1},  # It should have W pts.
        mops :: Any; # matrix operators
        ψ_t :: Float64,
        ψ_b :: Float64,
        d2ψ_t :: Float64,
        d2ψ_b :: Float64,
        d4ψ_t :: Float64,
        d4ψ_b :: Float64,
        f2    :: Float64,
        K2    :: Float64,
    )
       
        W_pts = length(RHS)

        B4 = zeros(Float64, W_pts)
        B2 = zeros(Float64, W_pts)
        B0 = zeros(Float64, W_pts)

        B4[1]   = d4ψ_t
        B4[end] = d4ψ_b

        B2[1]   = d2ψ_t
        B2[end] = d2ψ_b

        B0[1]   = ψ_t
        B0[end] = ψ_b

        
        # Derivation
        M = mops.M
        Δ = mops.Δ
        MΔ = mops.MΔ

        op_LHS = f2 * Δ * M + K2 * Δ * MΔ * MΔ * M
       

        op_LHS[1, 1]      = 1.0
        op_LHS[1, 2:end] .= 0.0
        
        op_LHS[end, end]      = 1.0
        op_LHS[end, 1:end-1] .= 0.0

        dropzeros!(op_LHS)

        RHS_adjust = RHS - K2 * Δ * ( MΔ * MΔ * B0 + MΔ * B2 + B4) - f2 * Δ * B0

        RHS_adjust[1] = B0[1]
        RHS_adjust[end] = B0[end]

        ψ = op_LHS \ RHS_adjust

        return ψ

    end

end

