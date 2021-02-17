struct LDS <: ENV
    state_size::Int64  # state_size
    action_size::Int64  # action_size
    x0::Array{Float64, 1}
    A::Array{Float64, 2}
    B::Array{Float64, 2}
    Q::Array{Float64, 2}
    R::Array{Float64, 2}
    H::Int64
end

function dynamics(model::LDS, x, u)
    model.A * x + model.B * u
end

function cost(model::LDS, x, u)
    x' * model.Q * x + u' * model.R * u
end

function reset!(model::LDS, x)
    x .= model.x0
end

function step(model::LDS, x, u)
    c = cost(model, x, u)
    xnew = dynamics(model, x, u)
    return xnew, c
end
