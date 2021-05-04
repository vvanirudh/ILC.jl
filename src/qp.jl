struct ILCData{n,m,T}
    Q::Diagonal{T,SVector{n,T}}
    R::Diagonal{T,SVector{m,T}}
    Qf::Diagonal{T,SVector{n,T}}
    Xref::Vector{SVector{n,T}}
    Uref::Vector{SVector{m,T}}
    times::Vector{T}
    u_bnd::T
end

function build_qp(data::ILCData, Xsim, Usim, A,B,K)
    Nh = length(data.Xref)
    n,m = size(B[1])
    Xref,Uref = data.Xref, data.Uref

    Np = (Nh-1)*(n+m)         # number of primals
    Nd = (Nh-1)*n + (Nh-1)*m  #  number of duals

    H = spzeros(Np,Np)
    q = zeros(Np)

    H = blockdiag(kron(I(Nh-2), blockdiag(sparse(data.R), sparse(data.Q))),
                  sparse(data.R), sparse(data.Qf))

    q = zeros((n+m)*(Nh-1))
    for k = 1:(Nh-2)
        q[(k-1)*(m+n) .+ (1:(m+n))] .= [0.0; data.Q*(Xsim[k+1]-Xref[k+1])]
    end
    q[(Nh-2)*(m+n) .+ (1:(m+n))] .= [0.0; data.Qf*(Xsim[Nh]-Xref[Nh])]

    C = spzeros(Nd,Np)
    lb = fill(-Inf,Nd)
    ub = fill(+Inf,Nd)
    #dynamics

    U = kron(I(Nh-1), [I zeros(m,n)]) #Matrix that picks out all u
    X = kron(I(Nh-1), [zeros(n,m) I]) #Matrix that picks out all x
    D = spzeros(n*(Nh-1), (n+m)*(Nh-1)) #dynamics constraints

    D[1:n,1:m] .= B[1]
    D[1:n,(m+1):(m+n)] .= -I(n)
    for k = 1:(Nh-2)
        D[(k*n).+(1:n), (m+(k-1)*(n+m)).+(1:(2*n+m))] .= [A[k+1]-B[k+1]*K[k+1] B[k+1] -I]

    end

    u_bnd = data.u_bnd
    for k=1:Nh-1
        sidx = (k-1) * n
        lb[sidx+1:sidx+n] .= zeros(n)
        ub[sidx+1:sidx+n] .= zeros(n)
    end
    for k=1:Nh-1
        sidx = n * (Nh-1) + (k-1) * m
        lb[sidx+1:sidx+m] .= -u_bnd .- Uref[k]
        ub[sidx+1:sidx+m] .= u_bnd .- Uref[k]
    end

    C = [D; U]

    # Build QP
    qp = OSQP.Model()
    OSQP.setup!(
        qp, P=H, q=q, A=C, l=lb, u=ub,
        # QP Parameters: feel free to change these, but these values should work fine
        eps_abs=1e-6, eps_rel=1e-6, eps_prim_inf = 1.0e-6, eps_dual_inf =
        1.0e-6, polish=1, verbose=0
    )
    return qp
end

function update_qp!(qp::OSQP.Model, data, X, U)
    Nh = length(data.Xref)
    n,m = length(X[1]), length(U[1])
    Qilc, Rilc, Qf = data.Q, data.R, data.Qf
    Xref,Uref = data.Xref, data.Uref

    Np = (Nh-1)*(n+m)
    q = zeros((n+m)*(Nh-1))
    for k = 1:(Nh-2)
        q[(k-1)*(m+n) .+ (1:(m+n))] .= [0.0; data.Q*(X[k+1]-Xref[k+1])]
    end
    q[(Nh-2)*(m+n) .+ (1:(m+n))] .= [0.0; data.Qf*(X[Nh]-Xref[Nh])]

    OSQP.update_q!(qp, q)
    return q
end
