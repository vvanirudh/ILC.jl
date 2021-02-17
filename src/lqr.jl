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
