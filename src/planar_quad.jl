struct PlanarQuadrotor <: ENV
    state_size::Int64
    action_size::Int64
    mass::Float64
    g::Float64
    ℓ::Float64
    J::Float64
    dt::Float64
    x0::Array{Float64, 1}
    xf::Array{Float64, 1}
    H::Int64
    wind::Float64
    function PlanarQuadrotor(mass::Float64, ℓ::Float64, x0::Array{Float64, 1},
                             xf::Array{Float64, 1}, H::Int64, wind::Float64)
        J = 0.2 * mass * ℓ^2
        new(6, 2, mass, 9.81, ℓ, J, 0.025, x0, xf, H, wind)
    end
end

function cont_dynamics(model::PlanarQuadrotor, x, u)

    mass, g, ℓ, J = model.mass, model.g, model.ℓ, model.J
    wind = model.wind
    posx, posy = x[1], x[2]

    θ = x[3]
    s, c = sincos(θ)
    ẍ = (1/mass) * (u[1] + u[2]) * s
    ÿ = (1/mass) * (u[1] + u[2]) * c - g
    θ̈ = (1/J) * (ℓ/2) * (u[2] - u[1])

    ẍ += wind * posx * 1.0
    ÿ += wind * posy * 1.0

    [x[4], x[5], x[6], ẍ, ÿ, θ̈]

end

function dynamics(model::PlanarQuadrotor, x, u)
    dt = model.dt
    # RK4
    x1 = cont_dynamics(model, x, u)
    x2 = cont_dynamics(model, x + 0.5 * dt * x1, u)
    x3 = cont_dynamics(model, x + 0.5 * dt * x2, u)
    x4 = cont_dynamics(model, x + dt * x3, u)

    xnew = x + (1/6) * dt * (x1 + 2*x2 + 2*x3 + x4)
    return xnew
end

function cost(model::PlanarQuadrotor, x, u)
    xf = model.xf
    uhover = [0.5 * model.mass * model.g, 0.5 * model.mass * model.g]

    Q = Diagonal([ones(3); fill(0.5, 3)])
    R = Diagonal(fill(1e-2, model.action_size))
    cost = (x - xf)' * Q * (x - xf)
    cost += (u - uhover)' * R * (u - uhover)
    # cost = (x[1] - xf[1])^2 + (x[2] - xf[2])^2
    # cost += (u[1] - 0.5 * model.mass * model.g)^2
    # cost += (u[2] - 0.5 * model.mass * model.g)^2
    # cost *= 10.0
    return cost
end

function step(model::PlanarQuadrotor, x, u)
    xnew = dynamics(model, x, u)
    c = cost(model, x, u)
    return xnew, c
end
