using PyPlot; const plt = PyPlot ; pygui(true)

# functions definition :
function solve(T0,t,KLG,KRG,FG,bc_gnodes,bc_val)
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

### PARAMETERS ###
## Parameters definition :
const κ = 1.0
const s = 0.0
const Tmax = 100.0
const σ = 1.0
const N_nodes = 2
const N_el = 1000
const N_gnodes = N_el*(N_nodes-1)+1

## Spatial and time domain :
const lx = 10.0
const x0 = -5
const Δx = lx/N_el

const Δt, lt = 0.01, 0.4
const x = x0:Δx:x0+lx
const t = 0:Δt:lt

### MAIN FUNCTION ###
function main_analytic()

    # Initial Temperature field
    get_T0_porte(x,val::Float64) = (0.3(lx-x0)<=x<=0.7(lx-x[1])) ? val : 0.0
    get_T0_exp(x,Tmax,σ) = Tmax*exp(-x^2/σ^2)
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

    # Local matrices building :
    MM = fill([ Δx/3 Δx/6 ; Δx/6 Δx/3 ],N_el)

    KM = κ_vec.*fill([1/Δx -1/Δx ; -1/Δx 1/Δx],N_el)
    F = s_vec.*fill([Δx/2, Δx/2],N_el)

    KL = (MM./Δt) + KM
    KR = MM./Δt

    # Global Matrices building
    KLG = zeros(Float64,(N_gnodes,N_gnodes))
    KRG = zeros(Float64,(N_gnodes,N_gnodes))
    FG = zeros(Float64,N_gnodes)

    for ielem in 1:N_el
        KLG[ielem:ielem+1,ielem:ielem+1] .+= KL[ielem]
        KRG[ielem:ielem+1,ielem:ielem+1] .+= KR[ielem]
        FG[ielem:ielem+1] .+= F[ielem]
    end

    # implement boundary condition on KLG :
    for ibc in eachindex(bc_gnodes)
        KLG[bc_gnodes[ibc],:] .= 0.0
        KLG[bc_gnodes[ibc],bc_gnodes[ibc]] = 1.0
    end

    # Solve over t :
    T = solve(T0,t,KLG,KRG,FG,bc_gnodes,bc_val)

    return x,T0,T
end

@time x,T0,T = main_analytic()
# Vizualize
plt.figure()
    plt.plot(x,T)
    plt.plot(x,T0,".g",markersize=1)

# plot against analytical solution with
analytical_sol(x,t,Tmax,σ,κ) = Tmax/(sqrt(1+(4*t*κ)/σ^2)) * exp(-x^2/(σ^2 + 4*t*κ))
real_sol = analytical_sol.(x,lt,Tmax,σ,κ[1])

plt.plot(x,real_sol,"--r",markersize=0.3)
