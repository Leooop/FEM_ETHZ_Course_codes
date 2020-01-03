using PyPlot; const plt = PyPlot ; pygui(true)
using Statistics
using LinearAlgebra
using SparseArrays

# functions definition :
function solve_once_chol(LG,FG,bc_eq_num,bc_val,force_vec_bc_addition)
    # Apply bc on rhs :
    b = FG .+ force_vec_bc_addition # add terms that allowed us to remove ibc columns
    for ibc in eachindex(bc_eq_num)
        b[bc_eq_num[ibc]] = bc_val[ibc] # set dirichlet boundary value
    end

    # Decompose the symmetric LG Matrix and solve :
    LGF = cholesky(Symmetric(LG)) # CHOLMOD sparse matrix cholesky decomposition
    # solve for u
    return LGF\b
end

function solve_once_lu(LG,FG,bc_eq_num,bc_val,force_vec_bc_addition)
    # Apply bc on rhs :
    b = FG .+ force_vec_bc_addition # add terms that allowed us to remove ibc columns
    for ibc in eachindex(bc_eq_num)
        b[bc_eq_num[ibc]] = bc_val[ibc] # set dirichlet boundary value
    end

    # Decompose the symmetric LG Matrix and solve :
    LGF = lu(LG) # UMFPACK sparse matrix LU factorisation
    # solve for u
    return LGF\b
end

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

"returns a (dof_per_node * N_gnodes) matrix associating the equation numbers with each global nodes"
function get_nf(dof_per_node,N_gnodes)
    return [ dof_per_node*(j-1)+1+(i-1) for i in 1:dof_per_node, j = 1:N_gnodes]
end

"returns the equation numbers associated to each element (columns)"
function get_g_g(el_gnodes,nf)
    nodes_per_el, N_el = size(el_gnodes)
    g_g = Matrix{Int}(undef,size(nf,1)*nodes_per_el,N_el)
    for iel in 1:N_el
        for inode in 1:nodes_per_el
            ind = 2*inode - 1
            g_g[ind:ind+1,iel] = nf[:,el_gnodes[inode,iel]]
        end
    end
    return g_g
end
"returns a vectorized (global nodes indexed) representation of a discrete matricial field"
grid2gnode(mat) = vec(reverse(mat',dims=2))

"returns a vectorized (global nodes indexed) representation of an arbitrary number of discrete matricial fields. with two input matrices u and v, it would return a vector [u1, v1, u2, v2, u3, ... , un, vn]"
function grid2gnode(mats...)
    len = length(mats)
    temp = repeat(mat1,inner=(1,len))
    for i = 2:len
        temp[:,i:len:end] .= mats[i]
    end
    return vec(reverse(temp',dims=2))
end

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

"Returns the jacobian matrix `∂x/∂ζ`"
function jaco(ζ,η,coords_elem_0L)
    return dN(ζ,η) * coords_elem_0L
end

"Builds arrays used to set boundary conditions in the main function"
function get_bc(bc::Dict,nf,gnodes_grid)
    bc_eq_num = Int[]
    bc_val = Float64[]
    for (key,value) in pairs(bc)

        if key == "top"
            bc_gnodes = gnodes_grid[1,:]
            normal_var = 2
            ortho_var = 1
        elseif key == "bot"
            bc_gnodes = gnodes_grid[end,:]
            normal_var = 2
            ortho_var = 1
        elseif key == "left"
            bc_gnodes = gnodes_grid[:,1]
            normal_var = 1
            ortho_var = 2
        elseif key == "right"
            bc_gnodes = gnodes_grid[:,end]
            normal_var = 1
            ortho_var = 2
        end

        len = length(bc_gnodes)
        value[2]!=0 && @error "Boundary conditions can only be of Dirichlet type for now"

        if value[1]==1
            append!(bc_eq_num,nf[normal_var,bc_gnodes])
            append!(bc_val,fill(value[3],len))
        elseif value[1]==0
            append!(bc_eq_num,nf[ortho_var,bc_gnodes])
            append!(bc_val,fill(0.0,len)) # No slip
            append!(bc_eq_num,nf[normal_var,bc_gnodes])
            append!(bc_val,fill(value[3],len))
        else
            @error "Free slip condition can only be set to 1 or 0"
        end
    end
    return bc_eq_num, bc_val
end

function get_strains_and_stresses(u,D,x_mat,y_mat,N_gnodesy)
    dx_mat = diff(x_mat,dims=2)
    dx_mat_centered = (dx_mat[2:end,:].+ dx_mat[1:end-1,:])./2
    dy_mat = abs.(diff(y_mat,dims=1))
    dy_mat_centered = (dy_mat[:,2:end].+ dy_mat[:,1:end-1])./2
    u_mat = gnode2grid(u[1:2:end],N_gnodesy)
    u_mat_centered = (u_mat[2:end,2:end] .+ u_mat[1:end-1,1:end-1])./2
    v_mat = gnode2grid(u[2:2:end],N_gnodesy)
    v_mat_centered = (v_mat[2:end,2:end] .+ v_mat[1:end-1,1:end-1])./2
    dudx = diff((u_mat[2:end,:] .+ u_mat[1:end-1,:])./2,dims=2)./dx_mat_centered
    dudy = diff((u_mat[:,2:end] .+ u_mat[:,1:end-1])./2,dims=1)./dy_mat_centered
    dvdx = diff((v_mat[2:end,:] .+ v_mat[1:end-1,:])./2,dims=2)./dx_mat_centered
    dvdy = diff((v_mat[:,2:end] .+ v_mat[:,1:end-1])./2,dims=1)./dy_mat_centered
    ϵ_mat = [dudx, dvdy, dudy.+dvdx]
    σxx_mat = zeros(Float64,size(dudx))
    σyy_mat = zeros(Float64,size(dudx))
    σxy_mat = zeros(Float64,size(dudx))
    for i in eachindex(σxx_mat)
        σxx_mat[i], σyy_mat[i], σxy_mat[i] = D*[ϵ_mat[1][i], ϵ_mat[2][i], ϵ_mat[3][i]]
    end
    σ_mat = [σxx_mat, σyy_mat, σxy_mat]
    return ϵ_mat, σ_mat
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
    r0 = zeros(Float64,dof_per_node*N_gnodes)

    # Vectorized parameters
    D = E/((1+ν)*(1-2ν)) * [ 1-ν   ν     0.0     ;
                              ν   1-ν    0.0     ;
                             0.0  0.0  0.5*(1-2ν)]
    D_vec = fill(D,N_el)
    ρ_vec = fill(ρ,N_el)

    #### Build Matrices used for the assembly :

    ## Matrix containing the global node number for node i and element j at index {ij} :
    gnodes_grid = get_gnodes_grid(N_gnodesx, N_gnodesy)
    el_gnodes = get_el_gnodes(N_gnodesx, N_gnodesy, N_nodes, N_elx, N_ely)
    # equation numbers for each gnode :
    nf = get_nf(dof_per_node,N_gnodes)
    # equation numbers for each element :
    g_g = get_g_g(el_gnodes,nf)

    # gnodes global coords :
    gnodes_coords = get_gnodes_coords(gnodes_grid,x_mat,y_mat,N_gnodes)

    ## Define boundary conditions
    bc_eq_num, bc_val = get_bc(bc::Dict,nf,gnodes_grid)
    # To which global nodes, on which variable, with which value ?
    # bc_gnodes_1 = [] #gnodes_grid[end:-1:1,1]
    # bc_var_1 = 1 # 1 : apply on u, 2 : apply on v
    # bc_val_1 = ubc
    # bc_gnodes_2 = gnodes_grid[1,:]#gnodes_grid[1:end,end]
    # bc_var_2 = 2 #apply on u
    # bc_val_2 = -ubc
    # bc_gnodes_3 = []#gnodes_grid[end,end-1:-1:2]
    # bc_var_3 = 1 #apply on v
    # bc_val_3 = 0.0
    # bc_gnodes_4 = gnodes_grid[end,:]#gnodes_grid[end,end-1:-1:2]
    # bc_var_4 = 2 #apply on v
    # bc_val_4 = ubc
    #
    # # build eq numbers and values ?
    # bc_eq_num = [nf[bc_var_1,bc_gnodes_1] ;
    #              nf[bc_var_2,bc_gnodes_2] ;
    #              nf[bc_var_3,bc_gnodes_3] ;
    #              nf[bc_var_4,bc_gnodes_4]]
    # bc_val = [fill(bc_val_1,length(bc_gnodes_1)) ;
    #           fill(bc_val_2,length(bc_gnodes_2)) ;
    #           fill(bc_val_3,length(bc_gnodes_3)) ;
    #           fill(bc_val_4,length(bc_gnodes_4))]

    #### Global arrays construction ####
    ## !!!! Assembly can be improved by only filling upper or lower triangular matrices of MM and KM since these are symmetric.
    KG_I = Int[]
    KG_J = Int[]
    KG_V = Float64[]
    FG = zeros(Float64,dof_per_node*N_gnodes)


    for ielem in 1:N_el
        # Initializing local arrays
        dof_per_el = N_nodes_per_el*dof_per_node
        KM = zeros(Float64,dof_per_el,dof_per_el)

        # Get global coordinates of current element nodes:
        nc = [gnodes_coords[:,el_gnodes[i,ielem]] for i in 1:N_nodes_per_el]
        coords_elem = [ 0.0 0.0 ;
                        nc[2][1]-nc[1][1] nc[2][2]-nc[1][2] ;
                        nc[3][1]-nc[1][1] nc[3][2]-nc[1][2] ;
                        nc[4][1]-nc[1][1] nc[4][2]-nc[1][2] ] # Careful the global coordinates are calculated relatively to the south-west node of the element, because of the definition of the shape functions !

        # Filling these local arrays
        invJip = 0.0
        for ip in 1:nip
            # Compute Jacobian at integration point and it's inverse and determinant at integration point
            Jip = jaco(ζᵢ[ip],ηᵢ[ip],coords_elem)
            invJip = inv(Jip)
            detJip = det(Jip)

            B̂ip = get_B̂(ζᵢ[ip],ηᵢ[ip],invJip)
            # KM matrix contribution from ip
            KM .+= B̂ip' * D_vec[ielem] * B̂ip .* (detJip*Wᵢ[ip])
            #println(maximum(abs.(KM.-KM')))
        end
        KM = Symmetric(KM) # Creating a symmetric view of KM since B'AB is symmetric as soon as A is symmetric, weirdly without this KM .- KM' returns some non zeros (the residual stay low relative the values though, abs.((KM .- KM')./KM) < 1e-15 )

        # Get new local arrays by grouping terms after finite difference time discretization
        #L = (MM./Δt) .+ KM
        #R = MM./Δt

        # increment the global arrays with the local ones

        for j in 1:dof_per_el
            for i in 1:dof_per_el
                push!(KG_I, g_g[i,ielem])
                push!(KG_J, g_g[j,ielem])
                push!(KG_V, KM[i,j])
                #push!(RG_I, el_gnodes[i,ielem])
                #push!(RG_J, el_gnodes[j,ielem])
                #push!(RG_V, R[i,j])
                #LG[el_gnodes[i,ielem],el_gnodes[j,ielem]] += L[i,j] # dense approach same for RG
            end
            #FG[el_gnodes[j,ielem]] += F[j]
        end
    end

    KG = sparse(KG_I,KG_J,KG_V,(N_gnodes*dof_per_node),(N_gnodes*dof_per_node))
    #RG = sparse(RG_I,RG_J,RG_V,N_gnodes,N_gnodes)

    ### implement boundary condition on LG : ###

    force_vec_bc_addition = zeros(Float64,(N_gnodes*dof_per_node)) # from schaum, removes the dependancy on Dirichlet nodes  when solving the livear system, and brings symmetry to the stiffness matrix

    # Start by removing all the stiffness matrix entries that must be set to zero to apply dirichlet bc.
    for ibc in eachindex(bc_eq_num)
        KG[bc_eq_num[ibc],:] .= 0.0
    end

    # Now for each dirichlet boundary node, we store the contribution of the nodal variable to the RHS at each relevant row to be able to substract T[ibc]*LG[j,ibc] to b[j] in the solve function, and set all non-diagonal terms of LG[:,ibc] to zero.
    for ibc in eachindex(bc_eq_num)
        force_vec_bc_addition .-= bc_val[ibc].*KG[:,bc_eq_num[ibc]] # save terms to be added to b later on
        KG[:,bc_eq_num[ibc]] .= 0.0 # remove ibc column
        KG[bc_eq_num[ibc],bc_eq_num[ibc]] = 1.0 # set the diagonal term to 1
    end


    #### Solve over t : ####
    if maximum(KG .- KG') <= 1e-10
        @info "using Cholesky decomposition"
        u = solve_once_chol(KG,FG,bc_eq_num,bc_val,force_vec_bc_addition)
    else
        @info "using lu factorization"
        u = solve_once_lu(KG,FG,bc_eq_num,bc_val,force_vec_bc_addition)
    end

    return u,D,KG
end

### PARAMETERS ###
## Parameters definition :
ρ = 8000.0 # acier : 8000
    E = 70e9
    ν = 0.499
    N_nodes = 2
    N_nodes_per_el = 4
    dof_per_node = 2

    ## Spatial and time domains :
    x0, y0 = 0.0, 0.0
    x_max, y_max = 10.0, 10.0
    lx, ly = x_max-x0, y_max-y0
    Δx_vals = [0.2,0.2,0.2]#[0.1, 0.05, 0.1]#[lx/50, lx/200, lx/50]
    Δy_vals = [0.2,0.2,0.2]#[0.1, 0.05, 0.1]#[ly/50, ly/200, ly/50]
    x_vals = [4,6]#[3, 7]
    y_vals = [4,6]#[3, 7]
    x_mat, y_mat = get_grid_coords(x0,y0,x_max,y_max,lx,ly,Δx_vals,Δy_vals,x_vals,y_vals, 3)
    #x_mat, y_mat = distort_grid(x_mat,y_mat,3) # uncomment to distort the grid
    Δx = diff(x_mat, dims=2)
    Δy = -diff(y_mat, dims=1)

    Δt, lt = 160.0, 100000.0
    t = 0:Δt:lt

    # boundary conditions :
    bc = Dict("top" => [1, 0, -0.001ly],
              "bot" => [1, 0, 0.0],
              "left" => [1, 0, -0.0005ly],
              "right" => [1, 0, 0.0005ly])

    ## Others Params
    N_elx, N_ely = size(x_mat,2)-1, size(y_mat,1)-1
    N_el = N_elx*N_ely
    dof_per_el = dof_per_node * N_nodes_per_el
    N_gnodesx = N_elx*(N_nodes-1)+1
    N_gnodesy = N_ely*(N_nodes-1)+1
    N_gnodes = N_gnodesx * N_gnodesy

    ### Choice of shape functions and numerical integration
    # Shape functions and their derivatives in terms of local and global variables
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

    dN1dx(ζ,η,invJip) = dN1dζ(ζ,η)*invJip[1,1] + dN1dη(ζ,η)*invJip[1,2]
    dN1dy(ζ,η,invJip) = dN1dζ(ζ,η)*invJip[2,1] + dN1dη(ζ,η)*invJip[2,2]
    dN2dx(ζ,η,invJip) = dN2dζ(ζ,η)*invJip[1,1] + dN2dη(ζ,η)*invJip[1,2]
    dN2dy(ζ,η,invJip) = dN2dζ(ζ,η)*invJip[2,1] + dN2dη(ζ,η)*invJip[2,2]
    dN3dx(ζ,η,invJip) = dN3dζ(ζ,η)*invJip[1,1] + dN3dη(ζ,η)*invJip[1,2]
    dN3dy(ζ,η,invJip) = dN3dζ(ζ,η)*invJip[2,1] + dN3dη(ζ,η)*invJip[2,2]
    dN4dx(ζ,η,invJip) = dN4dζ(ζ,η)*invJip[1,1] + dN4dη(ζ,η)*invJip[1,2]
    dN4dy(ζ,η,invJip) = dN4dζ(ζ,η)*invJip[2,1] + dN4dη(ζ,η)*invJip[2,2]

    function get_B̂(ζ,η,invJip)
        B1 = [dN1dx(ζ,η,invJip) 0.0 ;
              0.0 dN1dy(ζ,η,invJip) ;
              dN1dy(ζ,η,invJip) dN1dx(ζ,η,invJip)]
        B2 = [dN2dx(ζ,η,invJip) 0.0 ;
              0.0 dN2dy(ζ,η,invJip) ;
              dN2dy(ζ,η,invJip) dN2dx(ζ,η,invJip)]
        B3 = [dN3dx(ζ,η,invJip) 0.0 ;
              0.0 dN3dy(ζ,η,invJip) ;
              dN3dy(ζ,η,invJip) dN3dx(ζ,η,invJip)]
        B4 = [dN4dx(ζ,η,invJip) 0.0 ;
              0.0 dN4dy(ζ,η,invJip) ;
              dN4dy(ζ,η,invJip) dN4dx(ζ,η,invJip)]

        return [B1 B2 B3 B4]
    end

    N(ζ,η) = [N1(ζ,η), N2(ζ,η), N3(ζ,η), N4(ζ,η)] # Vector of shape functions
    dN(ζ,η) = [dN1dζ(ζ,η) dN2dζ(ζ,η) dN3dζ(ζ,η) dN4dζ(ζ,η) ;
            dN1dη(ζ,η) dN2dη(ζ,η) dN3dη(ζ,η) dN4dη(ζ,η)] # Matrix of shape functions derivatives with respect to local variables

# SOLVE :
@time u,D,KG = main_numerical();
    eps, sigma = get_strains_and_stresses(u,D,x_mat,y_mat,N_gnodesy)

# Vizualize #
# contourf
plt.figure()
    ax = plt.subplot()
    ax.axis("equal")
    u_mat = gnode2grid(u[1:2:end],N_gnodesy)
    v_mat = gnode2grid(u[2:2:end],N_gnodesy)
    plt.quiver(x_mat.+u_mat,y_mat.+v_mat,u_mat,v_mat, scale=1)
    plt.plot(x_mat.+u_mat,y_mat.+v_mat,".r",markersize=0.5)
    plt.plot(x_mat,y_mat,".b",markersize=0.5)

plt.figure()
    ax = plt.subplot()
    ax.axis("equal")
    x_center = (x_mat[2:end,2:end] .+ x_mat[1:end-1,1:end-1])./2
    y_center = (y_mat[2:end,2:end] .+ y_mat[1:end-1,1:end-1])./2
    plt.contourf(x_center,y_center,sigma[2]./1e6)
    plt.plot(x_mat,y_mat,".r",markersize=0.5)
    plt.colorbar()


#surface
plt.figure()
    plt.surf(x_mat,y_mat,gnode2grid(T,N_gnodesy))

# plot grid :
plt.figure()
    plt.plot(x_mat,y_mat,".r",markersize=1)

rmse(Y1,Y2) = sqrt(sum((Y1.-Y2).^2)/length(Y1))
