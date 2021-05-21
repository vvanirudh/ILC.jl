struct Cartpole <: ENV
    state_size::Int64
    action_size::Int64
    mc::Float64
    mp::Float64
    l::Float64
    g::Float64
    dt::Float64
    x0::Array{Float64, 1}
    H::Int64
    latency::Float64
    Cartpole(mc, mp, l, x0, H, latency) = new(4, 1, mc, mp, l, 9.81, 0.05, x0,
                                              H, latency)
end

function dynamics(model::Cartpole, x, u)
    mc = model.mc
    mp = model.mp
    l = model.l
    g = model.g
    dt = model.dt
    latency = model.latency

    q = x[1:2]
    qd = x[3:4]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp*g*l*s]
    B = [1, 0]

    qdd = -H\(C*qd + G - B*u[1])

    # Semi-implicit euler integration
    qdnew = qd + qdd * (dt - model.latency)
    qnew = q + qdnew * (dt - model.latency)

    return [qnew; qdnew]
end

function cost(model::Cartpole, x, u)
    function angle_normalize(θ)
        ((θ + pi) % (2*pi)) - pi
    end
    angle_normalize(x[2])^2 + u[1]^2 + x[1]^2
end

function reset!(model::Cartpole, x, u)
    x .= model.x0
end

function step(model::Cartpole, x, u)
    xnew = dynamics(model, x, u)
    c = cost(model, x, u)
    return xnew, c
end
