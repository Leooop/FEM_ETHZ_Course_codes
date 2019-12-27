using PyPlot; const plt = PyPlot ; pygui(true)
using Statistics
using LinearAlgebra
using SparseArrays

# functions definition :
function solve(T0,t,Δt,LG,RG,FG,bc_gnodes,bc_val)
    T = T0
    LGF = lu(LG) # UMFPACK sparse matrix LU factorisation
    b = RG*T .+ FG
    for i in eachindex(t)
        # form RHS
        @time mul!(b,RG,T) .+ FG
        # Apply bc on rhs :
        @time for ibc in eachindex(bc_gnodes)
            b[bc_gnodes[ibc]] = bc_val[ibc]
        end
        # solve for T
        println("Solving for T at iteration $i/$(length(t)), time = $(t[1]+(i-1)*Δt)")
        @time ldiv!(T,LGF,b)
    end
    return T
end

get_T0_exp_2D(x,y,x0,y0,Tmax,σ)  = [Tmax*exp(-((x[j]-x0)^2 + (y[i]-y0)^2)/σ^2) for i in eachindex(y), j in eachindex(x)]
#get_T0_porte(x,val::Float64) = (0.3(x_max-x0)<=x<=0.7(x_max-x[1])) ? val : 0.0

function get_el_gnodes(N_gnodesx, N_gnodesy, N_nodes)
    gnodes_grid = reverse([((j-1)*N_gnodesx).+i for j in 1:N_gnodesy, i in 1:N_gnodesx],dims=1)
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
    return gnodes_grid, el_gnodes
end

function get_gnodes_coords(gnodes_grid,x,y,N_gnodesx,N_gnodesy)
    # gives
    coords_grid = [(x[j],y[i]) for i in N_gnodesy:-1:1, j in 1:N_gnodesx]
    gnodes_coords = Matrix{Float64}(undef,2,N_gnodesx*N_gnodesy)
    for gnode in 1:(N_gnodesx*N_gnodesy)
        gnodes_coords[:,gnode] .= coords_grid[gnodes_grid.==gnode][1]
    end
    return gnodes_coords
end

grid2gnode(mat) = vec(reverse(mat',dims=2))
gnode2grid(vec, N_gnodesy) = reverse(reshape(vec,N_gnodesy,:),dims=2)'

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
    T0 = grid2gnode(get_T0_exp_2D(x,y,mean(x),mean(y),Tmax,σ))

    # Vectorized parameters
    D = diagm(fill(κ,2))
    s_vec = fill(s,N_el)
    ρ_vec = fill(ρ,N_el)
    cp_vec = fill(𝐶p,N_el)

    ## Matrix containing the global node number for node i and element j at index {ij} :
    gnodes_grid, gnodes_el = get_el_gnodes(N_gnodesx, N_gnodesy, N_nodes)

    ## Define boundary conditions
    # To which global nodes:
    bc_gnodes = [gnodes_grid[end:-1:1,1] ; gnodes_grid[1,2:end] ; gnodes_grid[2:end,end] ; gnodes_grid[end,end-1:-1:2]]
    # and its values
    bc_val = fill(0.0,length(bc_gnodes))

    #### Global arrays construction ####
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
        coords_elem_0L = [ 0.0 0.0 ; 0.0 Δy ; Δx Δy ; Δx 0.0 ]

        # Filling these local arrays
        for ip in 1:nip
            # Compute Jacobian at integration point and it's inverse and determinant at integration point
            Jip = jaco(ζᵢ[ip],ηᵢ[ip],coords_elem_0L)
            invJip = inv(Jip)
            detJip = det(Jip)

            # MM matrix contribution from ip
            MM .+= N(ζᵢ[ip],ηᵢ[ip]) * N(ζᵢ[ip],ηᵢ[ip])'
            #[ shape_funs[i](ζᵢ[ip],ηᵢ[ip])*shape_funs[j](ζᵢ[ip],ηᵢ[ip])*detJip*Wᵢ[ip] for i in 1:len, j in 1:len ]

            # KM matrix contribution from ip
            KM1 = (invJip * dN(ζᵢ[ip],ηᵢ[ip]))'
            KM2 = D * invJip * dN(ζᵢ[ip],ηᵢ[ip])
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
                push!(LG_I, gnodes_el[i,ielem])
                push!(LG_J, gnodes_el[j,ielem])
                push!(LG_V, L[i,j])
                push!(RG_I, gnodes_el[i,ielem])
                push!(RG_J, gnodes_el[j,ielem])
                push!(RG_V, R[i,j])
                #LG[gnodes_el[i,ielem],gnodes_el[j,ielem]] += L[i,j] # dense approach same for RG
                FG[gnodes_el[i,ielem]] += F[i]
            end
        end
    end

    LG = sparse(LG_I,LG_J,LG_V,N_gnodes,N_gnodes)
    RG = sparse(RG_I,RG_J,RG_V,N_gnodes,N_gnodes)

    # implement boundary condition on LG :
    for ibc in eachindex(bc_gnodes)
        LG[bc_gnodes[ibc],:] .= 0.0
        LG[bc_gnodes[ibc],bc_gnodes[ibc]] = 1.0
    end

    # Solve over t :
    T = solve(T0,t,Δt,LG,RG,FG,bc_gnodes,bc_val)

    return T0,T,LG
end

### PARAMETERS ###
## Parameters definition :
κ = 50.0   # acier : 50
    ρ = 8000.0 # acier : 8000
    𝐶p = 1000.0 # acier : 1000
    s = 0.0
    Tmax = 100.0
    σ = 1.0
    N_nodes = 2
    N_nodes_per_el = 4
    N_elx, N_ely = 200, 200
    N_el = N_elx*N_ely
    N_gnodesx = N_elx*(N_nodes-1)+1
    N_gnodesy = N_ely*(N_nodes-1)+1
    N_gnodes = N_gnodesx * N_gnodesy

    ## Spatial and time domains :
    lx, ly = 10.0, 10.0
    x0, y0 = 0.0, 0.0
    Δx, Δy = lx/N_elx, lx/N_ely

    Δt, lt = 1000000.0, 100000000.0
    x = x0:Δx:x0+lx
    y = y0:Δy:y0+ly
    t = 0:Δt:lt

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
# Vizualize
plt.figure()
    plt.surf(x,y,gnode2grid(T,N_gnodesy))
    #plt.surf(x,y,gnode2grid(T0,N_gnodesy))

rmse(Y1,Y2) = sqrt(sum((Y1.-Y2).^2)/length(Y1))
# plot against analytical solution with
