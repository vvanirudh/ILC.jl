function lqr(model::ENV, ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u)
    H = model.H
    n = model.state_size
    m = model.action_size

    k = zeros(H, m)
    K = zeros(H, m, n)
    V_x = zeros(n)
    V_xx = zeros(n, n)

    for h=H:-1:1
        ∇f_x, ∇f_u = ∇F_x[h, :, :], ∇F_u[h, :, :]
        ∇c_x, ∇c_u, ∇2c_x, ∇2c_u = ∇C_x[h, :], ∇C_u[h, :], ∇2C_x[h, :, :], ∇2C_u[h, :, :]

        Q_x = ∇c_x + ∇f_x' * V_x
        Q_u = ∇c_u + ∇f_u' * V_x

        Q_xx = ∇2c_x + ∇f_x' * V_xx * ∇f_x
        Q_ux = ∇f_u' * V_xx * ∇f_x
        Q_uu = ∇2c_u + ∇f_u' * V_xx * ∇f_u

        K[h, :, :] = -Q_uu\Q_ux
        k[h, :] = -Q_uu\Q_u

        V_x = Q_x - K[h, :, :]' * Q_uu * k[h, :]
        V_xx = Q_xx - K[h, :, :]' * Q_uu * K[h, :, :]
        V_xx = (V_xx + V_xx') / 2.0
    end
    return k, K
end

function tvlqr(A, B, Q, R, Qf)

    H = length(A) + 1
    n, m = size(B[1])
    P = [zeros(n, n) for k=1:H]
    K = [zeros(m, n) for k=1:H-1]

    P[end] .= Qf
    for k=reverse(1:H-1)
        K[k] .= (R + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
        P[k] .= Q + A[k]'P[k+1]*A[k] - A[k]'P[k+1]*B[k]*K[k]
    end
    return K, P
end

function tvilc(A, B, Ahat, Bhat, Q, R, Qf)

    H = length(A) + 1
    n, m = size(B[1])
    P = [zeros(n, n) for k=1:H]
    K = [zeros(m, n) for k=1:H-1]

    P[end] .= Qf
    for k=reverse(1:H-1)
        K[k] .= (R + Bhat[k]'P[k+1]*B[k]) \ (Bhat[k]'P[k+1]*A[k])
        # P[k] .= Q + K[k]'R * K[k] + (Ahat[k] - Bhat[k]*K[k])'P[k+1]*(A[k] - B[k]*K[k])
        P[k] .= Q + Ahat[k]'P[k+1] * A[k] - Ahat[k]'P[k+1]*B[k]*K[k]
    end
    return K, P
end
