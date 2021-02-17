struct Pendulum <: ENV
    state_size::Int64
    action_size::Int64
    H::Int64
    x0::Array{Float64, 1}
    dt::Float64
    max_torque::Float64
    max_speed::Float64
    mass::Float64
    ℓ::Float64
    Pendulum(H::Int64, x0::Array{Float64, 1}, mass::Float64) = new(2, 1, H, x0, 0.05, 8.0, 2.0, mass, 1.0)
end

function dynamics(model::Pendulum, x, u)
    θ, θ̇ = x[1], x[2]
    g = 10.0
    m = model.mass
    ℓ = model.ℓ
    dt = model.dt
    max_torque = model.max_torque
    max_speed = model.max_speed

    u_clipped = max.(min.(u, max_torque), -max_torque)
    
    θ̇new = θ̇ + (-3 * g / (2 * ℓ) * sin(θ + pi) + 3.0 / (m * ℓ^2) * u_clipped[1]) * dt
    θnew = θ + θ̇new * dt
    θ̇new = max.(min.(θ̇new, max_speed), -max_speed)

    return [θnew; θ̇new]
end

function cost(model::Pendulum, x, u)
    function angle_normalize(θ)
        ((θ + pi) % (2*pi)) - pi
    end
    angle_normalize(x[1])^2 + 0.1 * u[1]^2
end

function reset!(model::Pendulum, x, u)
    x .= model.x0
end

function step(model::Pendulum, x, u)
    xnew = dynamics(model, x, u)
    c = cost(model, x, u)
    return xnew, c
end
