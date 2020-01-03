using PyPlot; const plt = PyPlot ; pygui(true)
using Statistics
using LinearAlgebra
using SparseArrays

# functions definition :
"Solves for T at all times in t"
function solve_lu(T0,t,Δt,LG,RG,FG,bc_gnodes,bc_val,force_vec_bc_addition)
    T = T0
    LGF = lu(LG)
    # UMFPACK sparse matrix LU factorisation
    b = RG*T .+ FG
    for i in eachindex(t)

            # form RHS
            mul!(b,RG,T) .+ FG # in place matrix-vector multiplication to form b = RG*T
            # Apply bc on rhs :
            b += force_vec_bc_addition # add terms that allowed us to remove ibc columns
            for ibc in eachindex(bc_gnodes)
                b[bc_gnodes[ibc]] = bc_val[ibc] # set dirichlet boundary value
            end
            # solve for T
            (i%round(length(t)/100) == 0) && println("Solving for T at iteration $i/$(length(t)), time = $(t[1]+(i-1)*Δt)")
            ldiv!(T,LGF,b) # in place solving equivalent to out of place T = LGF\b. Accepts a factorized objet of LG

    end
    return T
end

function solve_chol(T0,t,Δt,LG,RG,FG,bc_gnodes,bc_val,force_vec_bc_addition)
    T = T0
    LGF = cholesky(Symmetric(LG))
    # UMFPACK sparse matrix LU factorisation
    b = RG*T .+ FG
    for i in eachindex(t)

            # form RHS
            mul!(b,RG,T) .+ FG # in place matrix-vector multiplication to form b = RG*T
            # Apply bc on rhs :
            b += force_vec_bc_addition # add terms that allowed us to remove ibc columns
            for ibc in eachindex(bc_gnodes)
                b[bc_gnodes[ibc]] = bc_val[ibc] # set dirichlet boundary value
            end
            # solve for T
            (i%round(length(t)/100) == 0) && println("Solving for T at iteration $i/$(length(t)), time = $(t[1]+(i-1)*Δt)")
            T = LGF\b

    end
    return T
end

"Compute a gaussian initial 2D temperature field"
get_T0_exp_2D(x_mat,y_mat,xc,yc,Tmax,σ)  = @. Tmax*exp(-((x_mat-xc)^2 + (y_mat-yc)^2)/σ^2)

"Returns a matrix containing the index of the global nodes"
get_gnodes_grid(N_gnodesx, N_gnodesy) = reverse([((j-1)*N_gnodesx).+i for j in 1:N_gnodesy, i in 1:N_gnodesx],dims=1)

"""
    get_el_gnodes(N_gnodesx, N_gnodesy, N_nodes, N_elx, N_ely)

Returns a matrix of dimension `(2*N_nodes, N_elx*N_ely)`, where N_nodes is the number of nodes per element face (2 with quadrilateral elements) and N_elx*N_ely is the total number of elements of the mesh. each `j` column of the returned matrix contains the indices of the global nodes defining the `jth element, following an antitrigonometric counting starting with the south-west node.

"""
function get_el_gnodes(N_gnodesx, N_gnodesy, N_nodes, N_elx, N_ely)
    gnodes_grid = get_gnodes_grid(N_gnodesx, N_gnodesy)
    el_gnodes = Matrix{Int}(undef,2*N_nodes,N_elx*N_ely)
    counter = 1
    for i = N_ely:-1:1
        for j = 1:N_elx
            el_global_nodes = gnodes_grid[i:i+1,j:j+1]
            el_gnodes[1,counter] = el_global_nodes[2,1]
            el_gnodes[2,counter] = el_global_nodes[1,1]
            el_gnodes[3,counter] = el_global_nodes[1,2]
            el_gnodes[4,counter] = el_global_nodes[2,2]
            counter +=1
        end
    end
    return el_gnodes
end

"returns a `2*N_gnodes matrix containing the cartesian 2D coordinates of all global nodes"
function get_gnodes_coords(gnodes_grid,x_mat,y_mat,N_gnodes)
    x_vec = grid2gnode(x_mat)
    y_vec = grid2gnode(y_mat)
    gnodes_coords = [x_vec y_vec]
    return gnodes_coords'
end

"returns a vectorized (global nodes indexed) representation of a discrete matricial field"
grid2gnode(mat) = vec(reverse(mat',dims=2))

"returns a vectorized (global nodes indexed) representation of a discrete matricial field"
gnode2grid(vec, N_gnodesy) = reverse(reshape(vec,N_gnodesy,:),dims=2)'

" returns the matrices containing the coordinates of each nodes"
function get_grid_coords(x0,y0,x_max,y_max,lx,ly,Δx_vals,Δy_vals,x_vals,y_vals, precision)
    x = Float64[x0]
    y = Float64[y0]
    count = 1
    while round(x[end],digits = precision) < x_max
        (count <= length(x_vals)) && (x[end] ≈ x_vals[count]) && (count += 1)
        push!(x,x[end]+Δx_vals[count])
    end
    count = 1
    while round(y[end],digits = precision) < y_max
        (count <= length(y_vals)) && (y[end] ≈ y_vals[count]) && (count += 1)
        push!(y,y[end]+Δy_vals[count])
    end
    x, y = round.(x,digits = precision), round.(y,digits = precision)
    x_mat = repeat(x,outer=(1,length(y)))'
    y_mat = reverse(repeat(y,outer=(1,length(x))),dims=1)
    return x_mat, y_mat
end

"returns distorded the `x_mat` and `y_mat` coordinates of the mesh by scaling each of those by the other one multiplied by `factor`"
function distort_grid(x_mat,y_mat,factor)
    x_mat_d = copy(x_mat)
    y_mat_d = copy(y_mat)
    y_mat_d[:,2:end] .= y_mat[:,2:end] .+ (x_mat[:,2:end]./maximum(x_mat)).*factor
    x_mat_d[1:end-1,:] .= x_mat[1:end-1,:] .+ (y_mat[1:end-1,:]./maximum(y_mat)).*factor
    return x_mat_d,y_mat_d
end

"Returns the jacobian matrix ``∂x/∂ζ evaluated as "
function jaco(ζ,η,coords_elem_0L)
    return dN(ζ,η) * coords_elem_0L
end

### MAIN FUNCTION ###
function main_numerical()

    # Gauss-Legendre quadrature parameters
    GL_order = 2
    nip = 4
    GL_points = (sqrt(1/3), -sqrt(1/3))
    GL_weights = (1.0,1.0)

    ζᵢ = [GL_points[1], GL_points[1], GL_points[2], GL_points[2]]
    ηᵢ = [GL_points[1], GL_points[2], GL_points[2], GL_points[1]]
    Wᵢ = vec([GL_weights[i]*GL_weights[j] for i in 1:2, j in 1:2])

    ### Initial conditions
    # Initial Temperature field
    T0 = grid2gnode(get_T0_exp_2D(x_mat,y_mat,mean(x_mat),mean(y_mat),Tmax,σ))

    # Vectorized parameters
    D_vec = fill(diagm([κx, κy]),N_el)
    s_vec = fill(s,N_el)
    ρ_vec = fill(ρ,N_el)
    cp_vec = fill(𝐶p,N_el)

    ## Matrix containing the global node number for node i and element j at index {ij} :
    gnodes_grid = get_gnodes_grid(N_gnodesx, N_gnodesy)
    el_gnodes = get_el_gnodes(N_gnodesx, N_gnodesy, N_nodes, N_elx, N_ely)

    # gnodes global coords :
    gnodes_coords = get_gnodes_coords(gnodes_grid,x_mat,y_mat,N_gnodes)

    ## Define boundary conditions
    # To which global nodes:
    bc_gnodes = [gnodes_grid[end:-1:1,1] ; gnodes_grid[1,2:end] ; gnodes_grid[2:end,end] ; gnodes_grid[end,end-1:-1:2]]
    # and its values
    bc_val = fill(0.0,length(bc_gnodes))

    #### Global arrays construction ####
    ## !!!! Assembly can be improved by only filling upper or lower triangular matrices of MM and KM since these are symmetric.
    LG_I = Int[]
    LG_J = Int[]
    LG_V = Float64[]
    RG_I = Int[]
    RG_J = Int[]
    RG_V = Float64[]
    FG = zeros(Float64,N_gnodes)


    for ielem in 1:N_el
        # Initializing local arrays
        len = N_nodes_per_el
        MM = zeros(Float64,len,len)
        KM = zeros(Float64,len,len)
        F = zeros(Float64,len)

        # Get global coordinates of current element nodes:
        nc = [gnodes_coords[:,el_gnodes[i,ielem]] for i in 1:N_nodes_per_el]
        coords_elem = [ 0.0 0.0 ;
                        nc[2][1]-nc[1][1] nc[2][2]-nc[1][2] ;
                        nc[3][1]-nc[1][1] nc[3][2]-nc[1][2] ;
                        nc[4][1]-nc[1][1] nc[4][2]-nc[1][2] ] # Careful the global coordinates are calculated relatively to the south-west node of the element, because of the definition of the shape functions !

        # Filling these local arrays
        for ip in 1:nip
            # Compute Jacobian at integration point and it's inverse and determinant at integration point
            Jip = jaco(ζᵢ[ip],ηᵢ[ip],coords_elem)
            invJip = inv(Jip)
            detJip = det(Jip)

            # MM matrix contribution from ip
            MM .+= N(ζᵢ[ip],ηᵢ[ip]) * N(ζᵢ[ip],ηᵢ[ip])' .* (detJip*Wᵢ[ip])
            #[ shape_funs[i](ζᵢ[ip],ηᵢ[ip])*shape_funs[j](ζᵢ[ip],ηᵢ[ip])*detJip*Wᵢ[ip] for i in 1:len, j in 1:len ]

            # KM matrix contribution from ip
            KM1 = (invJip * dN(ζᵢ[ip],ηᵢ[ip]))'
            KM2 = D_vec[ielem] * invJip * dN(ζᵢ[ip],ηᵢ[ip])
            KM .+= KM1 * KM2 .* (detJip*Wᵢ[ip])

            # F vector contribution from ip
            F .+= N(ζᵢ[ip],ηᵢ[ip]).*(detJip*Wᵢ[ip])
        end
        MM .*= ρ_vec[ielem]*cp_vec[ielem]
        F .*= s_vec[ielem]

        # Get new local arrays by grouping terms after finite difference time discretization
        L = (MM./Δt) .+ KM
        R = MM./Δt

        # increment the global arrays with the local ones

        for j in 1:len
            for i in 1:len
                push!(LG_I, el_gnodes[i,ielem])
                push!(LG_J, el_gnodes[j,ielem])
                push!(LG_V, L[i,j])
                push!(RG_I, el_gnodes[i,ielem])
                push!(RG_J, el_gnodes[j,ielem])
                push!(RG_V, R[i,j])
                #LG[el_gnodes[i,ielem],el_gnodes[j,ielem]] += L[i,j] # dense approach same for RG
            end
            FG[el_gnodes[j,ielem]] += F[j]
        end
    end

    LG = sparse(LG_I,LG_J,LG_V,N_gnodes,N_gnodes)
    RG = sparse(RG_I,RG_J,RG_V,N_gnodes,N_gnodes)

    ### implement boundary condition on LG : ###

    force_vec_bc_addition = zeros(Float64,N_gnodes) # from schaum, removes the dependancy on Dirichlet nodes  when solving the livear system, and brings symmetry to the stiffness matrix

    # Start by removing all the stiffness matrix entries that must be set to zero to apply dirichlet bc.
    for ibc in eachindex(bc_gnodes)
        LG[bc_gnodes[ibc],:] .= 0.0
    end

    # Now for each dirichlet boundary node, we store the contribution of the nodal variable to the RHS at each relevant row to be able to substract T[ibc]*LG[j,ibc] to b[j] in the solve function, and set all non-diagonal terms of LG[:,ibc] to zero.
    for ibc in eachindex(bc_gnodes)
        force_vec_bc_addition .-= bc_val[ibc].*LG[:,bc_gnodes[ibc]] # save terms to be added to b later on
        LG[:,bc_gnodes[ibc]] .= 0.0 # remove ibc column
        LG[bc_gnodes[ibc],bc_gnodes[ibc]] = 1.0 # set the diagonal term to 1
    end


    #### Solve over t : ####
    if maximum(LG .- LG') <= 1e-10
        T = solve_chol(T0,t,Δt,LG,RG,FG,bc_gnodes,bc_val,force_vec_bc_addition)
    else
        T = solve_lu(T0,t,Δt,LG,RG,FG,bc_gnodes,bc_val,force_vec_bc_addition)
    end

    return T0,T,LG
end

### PARAMETERS ###
## Parameters definition :
κx = 50.0   # acier : 50
    κy = 50.0
    ρ = 8000.0 # acier : 8000
    𝐶p = 1000.0 # acier : 1000
    s = 0.0
    Tmax = 100.0
    σ = 1.0
    N_nodes = 2
    N_nodes_per_el = 4

    ## Spatial and time domains :
    x0, y0 = 0.0, 0.0
    x_max, y_max = 10.0, 10.0
    lx, ly = x_max-x0, y_max-y0
    Δx_vals = [0.1,0.1,0.1]#[0.1, 0.05, 0.1]#[lx/50, lx/200, lx/50]
    Δy_vals = [0.1,0.1,0.1]#[0.1, 0.05, 0.1]#[ly/50, ly/200, ly/50]
    x_vals = [4,6]#[3, 7]
    y_vals = [4,6]#[3, 7]
    x_mat, y_mat = get_grid_coords(x0,y0,x_max,y_max,lx,ly,Δx_vals,Δy_vals,x_vals,y_vals, 3)
    x_mat, y_mat = distort_grid(x_mat,y_mat,3) # uncomment to distort the grid
    Δx = diff(x_mat, dims=2)
    Δy = -diff(y_mat, dims=1)

    Δt, lt = 160.0, 100000.0
    t = 0:Δt:lt

    ## Others Params
    N_elx, N_ely = size(x_mat,2)-1, size(y_mat,1)-1
    N_el = N_elx*N_ely
    N_gnodesx = N_elx*(N_nodes-1)+1
    N_gnodesy = N_ely*(N_nodes-1)+1
    N_gnodes = N_gnodesx * N_gnodesy

    ### Choice of shape functions and numerical integration
    # Shape functions and their derivatives in terms of local variables
    N1(ζ,η) = (1/4)*(1-ζ)*(1-η)
    N2(ζ,η) = (1/4)*(1-ζ)*(1+η)
    N3(ζ,η) = (1/4)*(1+ζ)*(1+η)
    N4(ζ,η) = (1/4)*(1+ζ)*(1-η)

    dN1dζ(ζ,η) = -(1/4)*(1-η)
    dN1dη(ζ,η) = -(1/4)*(1-ζ)
    dN2dζ(ζ,η) = -(1/4)*(1+η)
    dN2dη(ζ,η) = (1/4)*(1-ζ)
    dN3dζ(ζ,η) = (1/4)*(1+η)
    dN3dη(ζ,η) = (1/4)*(1+ζ)
    dN4dζ(ζ,η) = (1/4)*(1-η)
    dN4dη(ζ,η) = -(1/4)*(1+ζ)

    N(ζ,η) = [N1(ζ,η), N2(ζ,η), N3(ζ,η), N4(ζ,η)] # Vector of shape functions
    dN(ζ,η) = [dN1dζ(ζ,η) dN2dζ(ζ,η) dN3dζ(ζ,η) dN4dζ(ζ,η) ;
            dN1dη(ζ,η) dN2dη(ζ,η) dN3dη(ζ,η) dN4dη(ζ,η)] # Matrix of shape functions derivatives with respect to local variables

# SOLVE :
@time T0,T,LG = main_numerical();

# Vizualize #
# contourf
plt.figure()
    ax = plt.subplot()
    ax.axis("equal")
    plt.contourf(x_mat,y_mat,gnode2grid(T,N_gnodesy))
    plt.plot(x_mat,y_mat,".r",markersize=0.5)

#surface
plt.figure()
    plt.surf(x_mat,y_mat,gnode2grid(T,N_gnodesy))

# plot grid :
plt.figure()
    plt.plot(x_mat,y_mat,".r",markersize=1)

rmse(Y1,Y2) = sqrt(sum((Y1.-Y2).^2)/length(Y1))
