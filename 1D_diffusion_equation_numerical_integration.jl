using PyPlot; const plt = PyPlot ; pygui(true)

# functions definition :
function solve(T0,t,Δt,KLG,KRG,FG,bc_gnodes,bc_val)
    T = T0
    for i in eachindex(t)
        # form RHS
        b = KRG*T .+ FG
        # Apply bc on rhs :
        for ibc in eachindex(bc_gnodes)
            b[bc_gnodes[ibc]] = bc_val[ibc]
        end
        # solve for T
        println("Solving for T at iteration $i/$(length(t)), time = $(t[1]+(i-1)*Δt)")
        T = KLG\b
    end
    return T
end

get_T0_porte(x,val::Float64) = (0.3(lx-x0)<=x<=0.7(lx-x[1])) ? val : 0.0
get_T0_exp(x,Tmax,σ) = Tmax*exp(-x^2/σ^2)

### PARAMETERS ###
## Parameters definition :
const κ = 1.0
const s = 0.0
const Tmax = 100.0
const σ = 1.0
const N_nodes = 2
const N_el = 1000
const N_gnodes = N_el*(N_nodes-1)+1

## Spatial and time domains :
const lx = 10.0
const x0 = -5
const Δx = lx/N_el

const Δt, lt = 0.01, 0.4
const x = x0:Δx:x0+lx
const t = 0:Δt:lt

### MAIN FUNCTION ###
function main_numerical()

    ### Choice of shape functions and numerical integration
    # Shape functions and their derivatives in terms of the local variable
    N1(ζ) = (1/2)*(1-ζ)
    N2(ζ) = (1/2)*(1+ζ)
    dN1dζ(ζ) = -(1/2)
    dN2dζ(ζ) = 1/2

    # Gauss-Legendre quadrature parameters
    GL_order = 2
    GL_points = (sqrt(1/3), -sqrt(1/3))
    GL_weights = (1.0,1.0)

    #Evaluation and storage of the shapes functions evaluated at
    #ζ = GL_points, as well as their derivatives
    N1_GL_points = (N1(GL_points[1]), N1(GL_points[2]))
    N2_GL_points = (N2(GL_points[1]), N2(GL_points[2]))
    dN1dζ_GL_points = (dN1dζ(GL_points[1]), dN1dζ(GL_points[2]))
    dN2dζ_GL_points = (dN2dζ(GL_points[1]), dN2dζ(GL_points[2]))

    # Jacobian and its inverse
    dxdζ = Δx/2
    dζdx = 2/Δx

    ### Initial conditions
    # Initial Temperature field
    T0 = get_T0_exp.(x,Tmax,σ)

    # variable diffusivity in space
    κ_vec = κ.+zeros(Float64,N_el)
    #κ[1:10] .= 1.0
    #κ[end-9:end] .= 1.0

    # Variable source
    s_vec = s.+zeros(Float64,N_el)
    #s[Int(round(N_el/2)-20):Int(round(N_el/2)+20)] .= 10.0


    ## Matrix containing the global node number for node i and element j at index {ij} :
    gnode_mat = [(j + (i - 1)) for i in 1:2, j in 1:N_el ]

    ## Define boundary conditions
    # To which global nodes:
    bc_gnodes = (1, N_gnodes)
    bc_val = (0.0,0.0)

    #### Global arrays construction ####
    KLG = zeros(Float64,(N_gnodes,N_gnodes))
    KRG = zeros(Float64,(N_gnodes,N_gnodes))
    FG = zeros(Float64,N_gnodes)

    for ielem in 1:N_el
        # Initializing local arrays
        MM = zeros(Float64,N_nodes,N_nodes)
        KM = zeros(Float64,N_nodes,N_nodes)
        F = zeros(Float64,N_nodes)

        # Filling these local arrays
        for ip in 1:length(GL_points)
            MM .+= [ N1_GL_points[ip]*N1_GL_points[ip]*dxdζ*GL_weights[ip]                             N1_GL_points[ip]*N2_GL_points[ip]*dxdζ*GL_weights[ip] ;
            N1_GL_points[ip]*N2_GL_points[ip]*dxdζ*GL_weights[ip] N2_GL_points[ip]*N2_GL_points[ip]*dxdζ*GL_weights[ip] ]

            KM .+= [ dN1dζ_GL_points[ip]*dN1dζ_GL_points[ip]*dζdx^2*dxdζ*GL_weights[ip] dN1dζ_GL_points[ip]*dN2dζ_GL_points[ip]*dζdx^2*dxdζ*GL_weights[ip] ;
            dN1dζ_GL_points[ip]*dN2dζ_GL_points[ip]*dζdx^2*dxdζ*GL_weights[ip] dN2dζ_GL_points[ip]*dN2dζ_GL_points[ip]*dζdx^2*dxdζ*GL_weights[ip] ]

            F  .+= [ N1_GL_points[ip]*dxdζ*GL_weights[ip], N2_GL_points[ip]*dxdζ*GL_weights[ip] ]
        end
        KM .*= κ_vec[ielem]
        F .*= s_vec[ielem]

        # Get new local arrays by grouping terms after finite difference time integration
        KL = (MM./Δt) + KM
        KR = MM./Δt

        # increment the global arrays with the local ones
        KLG[ielem:ielem+1,ielem:ielem+1] .+= KL
        KRG[ielem:ielem+1,ielem:ielem+1] .+= KR
        FG[ielem:ielem+1] .+= F
    end


    # implement boundary condition on KLG :
    for ibc in eachindex(bc_gnodes)
        KLG[bc_gnodes[ibc],:] .= 0.0
        KLG[bc_gnodes[ibc],bc_gnodes[ibc]] = 1.0
    end

    # Solve over t :
    T = solve(T0,t,Δt,KLG,KRG,FG,bc_gnodes,bc_val)

    return x,T0,T
end

@time x,T0,T = main_numerical()
# Vizualize
plt.figure()
    plt.plot(x,T)
    plt.plot(x,T0,".g",markersize=1)

# plot against analytical solution with
analytical_sol(x,t,Tmax,σ,κ) = Tmax/(sqrt(1+(4*t*κ)/σ^2)) * exp(-x^2/(σ^2 + 4*t*κ))
real_sol = analytical_sol.(x,lt,Tmax,σ,κ[1])

plt.plot(x,real_sol,"--r",markersize=0.3)
