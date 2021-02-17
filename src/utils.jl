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
