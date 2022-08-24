
module MatrixOperatorModule

    using SparseArrays

    export MatrixOperators, speye

    @inline function speye(dtype, n)
        return spdiagm(0=>ones(dtype, n))
    end

    # Assuming x-direction is periodic
    struct MatrixOperators

        Nz
        Ny
        Nx

        T_dim
        V_dim
        W_dim
        VW_dim
        U_dim

        T_pts
        V_pts
        W_pts
        VW_pts
        U_pts

        # Nomenclature:
        #
        # [new-grid][direction][old-grid]
        #
        # U_W_T : sending variable westward from T grid to U grid

        T_I_T
        V_I_V
        W_I_W
        VW_I_VW
        U_I_U

        V_S_T
        V_N_T
        T_S_V
        T_N_V
       
        T_N_T
        T_S_T

        # === [ Begin mirror operator ] ===
        
        V_mS_T
        V_mN_T

        # These two operators are the same as T_S_V and T_N_V
        # T_mS_V
        # T_mN_V
       
        T_mN_T
        T_mS_T
        
        # === [ End mirror operator ] ===


        T_UP_T
        T_DN_T
     
        T_UP_W
        T_DN_W

        W_UP_T
        W_DN_T

        VW_N_W
        VW_S_W
        W_N_VW
        W_S_VW

        VW_UP_V
        VW_DN_V
        V_UP_VW
        V_DN_VW

        U_W_T
        U_E_T
        U_W_U
        U_E_U
        T_W_U
        T_E_U

        T_E_T
        T_W_T
        
        function MatrixOperators(;
            Ny             :: Int64,
            Nz             :: Int64,
            Nx             :: Int64,
        )
            
            # Making operator
            T_dim  =  (Nz,   Ny   , Nx)
            V_dim  =  (Nz,   Ny+1 , Nx)
            W_dim  =  (Nz+1, Ny   , Nx)
            VW_dim =  (Nz+1, Ny+1 , Nx)
            U_dim  =  (Nz,   Ny   , Nx+1)

            T_pts  = reduce(*, T_dim)
            V_pts  = reduce(*, V_dim)
            W_pts  = reduce(*, W_dim)
            VW_pts = reduce(*, VW_dim)
            U_pts  = reduce(*, U_dim)

            T_I_T   = speye(Float64, T_pts)
            V_I_V   = speye(Float64, V_pts)
            W_I_W   = speye(Float64, W_pts)
            VW_I_VW = speye(Float64, VW_pts)
            U_I_U   = speye(Float64, U_pts)

            T_I_T_expand   = vcat(T_I_T,   zeros(Float64, 1, T_pts))
            V_I_V_expand   = vcat(V_I_V,   zeros(Float64, 1, V_pts))
            W_I_W_expand   = vcat(W_I_W,   zeros(Float64, 1, W_pts))
            VW_I_VW_expand = vcat(VW_I_VW, zeros(Float64, 1, VW_pts))
            U_I_U_expand   = vcat(U_I_U,   zeros(Float64, 1, U_pts))

            num_T  = zeros(Int64, T_dim...)
            num_V  = zeros(Int64, V_dim...)
            num_W  = zeros(Int64, W_dim...)
            num_VW = zeros(Int64, VW_dim...)
            num_U  = zeros(Int64, U_dim...)

            num_T[:]  = 1:length(num_T)
            num_V[:]  = 1:length(num_V)
            num_W[:]  = 1:length(num_W)
            num_VW[:] = 1:length(num_VW)
            num_U[:]  = 1:length(num_U)

            T   = num_T * 0
            V   = num_V * 0
            W   = num_W * 0
            VW  = num_VW * 0
            U   = num_U * 0


            #smb = SparseMatrixBuilder(Nx*(Ny+1)*(Nz+1)*4)
            function build!(id_mtx, idx; wipe=:none)
               #println("Build!")
                local result
                rows = size(id_mtx)[1]
                if wipe == :n
                    idx[:, end, :] .= rows
                elseif wipe == :s
                    idx[:, 1,   :] .= rows
                elseif wipe == :t
                    idx[1, :,   :] .= rows
                elseif wipe == :b
                    idx[end, :, :] .= rows
                elseif wipe == :e
                    idx[:, :, end] .= rows
                elseif wipe == :w
                    idx[:, :, 1  ] .= rows
                elseif wipe != :none
                    throw(ErrorException("Wrong keyword"))
                end
               
                # using transpose speeds up by 100 times 
                tp = transpose(id_mtx) |> sparse
                result = transpose(tp[:, view(idx, :)]) |> sparse
                #result = id_mtx[view(idx, :), :]
                #dropzeros!(result)

                idx .= 0 # clean so that debug is easir when some girds are not assigned
                return result
            end


            # north and south passing mtx

            T[:, :, :] = num_V[:, 2:Ny+1, :];    T_S_V = build!(V_I_V_expand, T);
            T[:, :, :] = num_V[:, 1:Ny,   :];    T_N_V = build!(V_I_V_expand, T);

            V[:, 1:Ny, :]    = num_T;            V_S_T  = build!(T_I_T_expand, V; wipe=:n)
            V[:, 2:Ny+1, :]  = num_T;            V_N_T  = build!(T_I_T_expand, V; wipe=:s)

            # T to T operators
            T_N_T = T_N_V * V_N_T
            T_S_T = T_S_V * V_S_T

            # "mirror" north and south passing mtx
            if Nx == 2
                
                println("Mirror Condition: East and west boundaries share the same northern and southern edge.")

                V[:, 1:Ny, :]    = num_T
                V[:, Ny+1, 1]    = num_T[:, Ny, 2]
                V[:, Ny+1, 2]    = num_T[:, Ny, 1]
                V_mS_T  = build!(T_I_T_expand, V; wipe=:none)


                V[:, 2:Ny+1, :]  = num_T
                V[:, 1, 1] = num_T[:,  1, 2]
                V[:, 1, 2] = num_T[:,  1, 1]
                V_mN_T  = build!(T_I_T_expand, V; wipe=:none)

                # T to T operators
                T_mN_T = T_N_V * V_mN_T
                T_mS_T = T_S_V * V_mS_T

            else

                V_mS_T = copy(V_S_T)
                V_mN_T = copy(V_N_T)
                T_mS_T = copy(T_S_T)
                T_mN_T = copy(T_N_T)

            end


            # VW grid north and south

            #VW[:, 1:Ny, :]   = num_W;                VW_S_W = build!(W_I_W_expand, VW; wipe=:n)
            #VW[:, 2:Ny+1, :] = num_W;                VW_N_W = build!(W_I_W_expand, VW; wipe=:s)

            #W_N_VW = VW_S_W' |> sparse
            #W_S_VW = VW_N_W' |> sparse
            
            W[:, :, :] = num_VW[:, 2:Ny+1, :];    W_S_VW = build!(VW_I_VW_expand, W; wipe=:none)
            W[:, :, :] = num_VW[:, 1:Ny, :];      W_N_VW = build!(VW_I_VW_expand, W; wipe=:none)

            VW_N_W = W_S_VW' |> sparse
            VW_S_W = W_N_VW' |> sparse

            # upward, downward passing mtx
            T[1:Nz-1, :, :] = view(num_T, 2:Nz, :, :);    T_UP_T = build!(T_I_T_expand, T; wipe=:b)
            T[2:Nz, :, :] = view(num_T, 1:Nz-1, :, :);    T_DN_T = build!(T_I_T_expand, T; wipe=:t)

            T[:, :, :] = view(num_W, 2:Nz+1, :, :);       T_UP_W = build!(W_I_W_expand, T)
            T[:, :, :] = view(num_W, 1:Nz, :, :);         T_DN_W = build!(W_I_W_expand, T)

            V[:, :, :] = view(num_VW, 2:Nz+1, :, :);      V_UP_VW = build!(VW_I_VW_expand, V)
            V[:, :, :] = view(num_VW, 1:Nz  , :, :);      V_DN_VW = build!(VW_I_VW_expand, V)

            # inverse directions
            W_DN_T = T_UP_W' |> sparse
            W_UP_T = T_DN_W' |> sparse
            
            VW_UP_V = V_DN_VW' |> sparse
            VW_DN_V = V_UP_VW' |> sparse

            # east west passing mtx
            U[:, :, 1:Nx] = num_T;       U_W_T = build!(T_I_T_expand, U; wipe=:e)
            U[:, :, 2:Nx+1] = num_T;       U_E_T = build!(T_I_T_expand, U; wipe=:w)

            T[:, :, :] = num_U[:, :, 2:Nx+1];       T_W_U = build!(U_I_U_expand, T)
            T[:, :, :] = num_U[:, :, 1:Nx  ];       T_E_U = build!(U_I_U_expand, T)


            U_W_U = U_W_T * T_W_U
            U_E_U = U_E_T * T_E_U

            T_W_T = T_W_U * U_W_T
            T_E_T = T_E_U * U_E_T



            return new(
                Nz,
                Ny,
                Nx,

                T_dim,
                V_dim,
                W_dim,
                VW_dim,
                U_dim,


                T_pts,
                V_pts,
                W_pts,
                VW_pts,
                U_pts,
                
                T_I_T,
                V_I_V,
                W_I_W,
                VW_I_VW,
                U_I_U,

                V_S_T,
                V_N_T,
                T_S_V,
                T_N_V,
               
                T_N_T,
                T_S_T,

                V_mS_T,
                V_mN_T,
               
                T_mN_T,
                T_mS_T,

                T_UP_T,
                T_DN_T,
             
                T_UP_W,
                T_DN_W,

                W_UP_T,
                W_DN_T,

                VW_N_W,
                VW_S_W,
                W_N_VW,
                W_S_VW,

                VW_UP_V,
                VW_DN_V,
                V_UP_VW,
                V_DN_VW,

                U_W_T,
                U_E_T,
                U_W_U,
                U_E_U,
                T_W_U,
                T_E_U,
     
                T_E_T,
                T_W_T,

            )

        end
    end

end
