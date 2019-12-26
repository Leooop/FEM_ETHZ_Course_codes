L = 1.0
coords_elem_0L = [0.0 0.0 ;
                         0.0 L ;
                         L L ;
                         L 0.0] # square
coords_elem_0L = [0.0 0.0 ;
                         0.0 L ;
                         L 2*L ;
                         L 0.0] # cut rectangle
coords_elem_0L = [0.0 0.0 ;
                  0.5 L ;
                  L L ;
                  0.5 0.0] # parallelogram


function jaco(ζ,η,coords_elem_0L)
    return dN(ζ,η) * coords_elem_0L
end
# Gauss-Legendre quadrature parameters
GL_order = 2
nip = 4
GL_points = (sqrt(1/3), -sqrt(1/3))
GL_weights = (1.0,1.0)

ζᵢ = [GL_points[1], GL_points[1], GL_points[2], GL_points[2]]
ηᵢ = [GL_points[1], GL_points[2], GL_points[2], GL_points[1]]
Wᵢ = vec([GL_weights[i]*GL_weights[j] for i in 1:2, j in 1:2])

### Choice of shape functions and numerical integration
# Shape functions and their derivatives in terms of the local variable
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

integral = 0.0
for ip = 1:4
    # Jacobian and associated terms
    Jip = jaco(ζᵢ[ip],ηᵢ[ip],coords_elem_0L)
    invJip = inv(Jip)
    detJip = det(Jip)

    global integral += detJip*Wᵢ[ip]
end
integral
