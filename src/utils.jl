function linearize_dynamics(model::ENV, x, u)
    ∇f_x = ForwardDiff.jacobian(x -> dynamics(model, x, u), x)
    ∇f_u = ForwardDiff.jacobian(u -> dynamics(model, x, u), u)
    return ∇f_x, ∇f_u
end

function quadraticize_cost(model::ENV, x, u)
    ∇c_x = ForwardDiff.gradient(x -> cost(model, x, u), x)
    ∇c_u = ForwardDiff.gradient(u -> cost(model, x, u), u)
    ∇2c_x = ForwardDiff.hessian(x -> cost(model, x, u), x)
    ∇2c_u = ForwardDiff.hessian(u -> cost(model, x, u), u)
    return ∇c_x, ∇c_u, ∇2c_x, ∇2c_u
end

function linearize_quadraticize_traj(model::ENV, X, U)
    H = model.H
    n = model.state_size
    m = model.action_size
    ∇F_x = zeros(H, n, n)
    ∇F_u = zeros(H, n, m)
    ∇C_x = zeros(H, n)
    ∇C_u = zeros(H, m)
    ∇2C_x = zeros(H, n, n)
    ∇2C_u = zeros(H, m, m)

    for h=1:H
        ∇f_x, ∇f_u = linearize_dynamics(model, X[h, :], U[h, :])
        ∇F_x[h, :, :] .= ∇f_x
        ∇F_u[h, :, :] .= ∇f_u
        ∇c_x, ∇c_u, ∇2c_x, ∇2c_u = quadraticize_cost(model, X[h, :], U[h, :])
        ∇C_x[h, :] .= ∇c_x
        ∇C_u[h, :] .= ∇c_u
        ∇2C_x[h, :, :] .= ∇2c_x
        ∇2C_u[h, :, :] .= ∇2c_u
    end
    return ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u
end

function rollout(env::ENV, U_old; k=nothing, K=nothing, X_old=nothing, alpha=1.0)
    H = env.H
    n = env.state_size
    m = env.action_size
    X = zeros(H+1, n)
    U = zeros(H+1, m)

    X[1, :] = env.x0
    cost = 0.
    for h=1:H
        if k == nothing
            U[h, :] = U_old[h, :]
        else
            U[h, :] = U_old[h, :] + alpha * k[h, :] + K[h, :, :] * (X[h, :] - X_old[h, :])
        end
        X[h+1, :], instant_cost = step(env, X[h, :], U[h, :])
        cost += instant_cost
    end
    return X, U, cost
end

function linearize_around_ref(model::Acrobot, Xref, Uref, times)
    N = length(times)
    n = length(Xref[1])
    m = length(Uref[1])
    A = [zeros(n,n) for k = 1:N-1]
    B = [zeros(n,m) for k = 1:N-1]
    ∇f = RobotDynamics.DynamicsJacobian(model)
    for k = 1:N-1
        local t = times[k]
        local dt = times[k+1] - times[k]
        z = KnotPoint(SVector{n}(Xref[k]), SVector{m}(Uref[k]), dt, t)
        discrete_jacobian!(RK4, ∇f, model, z)
        A[k] .= ∇f.A
        B[k] .= ∇f.B
    end
    return A, B
end

function lqr_cost(Q, R, Qf, X, U, N)
    J = 0.0
    for k=1:N-1
        J += 0.5 * X[k]' * Q * X[k] + 0.5 * U[k]' * R * U[k]
    end
    J += 0.5 * X[end]' * Qf * X[end]
    return J
end

function rollout(model::Acrobot, K, Xref, Uref, times; u_bnd=20.0, x0=Xref[1])
    n,m = size(model)
    N = length(K) + 1
    X = [@SVector zeros(n) for k = 1:N]
    U = [@SVector zeros(m) for k = 1:N-1]
    X[1] = x0

    for k=1:N-1
        dt = times[k+1] - times[k]
        U[k] = Uref[k] - K[k] * (X[k] - Xref[k])
        U[k] = min.(u_bnd, max.(-u_bnd, U[k]))
        z = KnotPoint(X[k], U[k], dt)
        X[k+1] = discrete_dynamics(RK4, model, z)
    end

    return X, U
end

function rollout(x0, A, B, Q, R, Qf, H, K)
    n = length(x0)
    m = size(B, 2)

    X = [zeros(n) for i=1:H+1]
    U = [zeros(m) for i=1:H]

    X[1] .= copy(x0)
    c = 0.
    for k=1:H
        U[k] .= -K[k] * X[k]
        X[k+1] .= A * X[k] + B * U[k]
        c = c + X[k]' * Q * X[k] + U[k]' * R * U[k]
    end
    c = c + X[H+1]' * Qf * X[H+1]

    return X, U, c
end
