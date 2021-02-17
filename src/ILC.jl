module ILC

using Plots
using PyPlot
using ForwardDiff
using LinearAlgebra
using Printf

include("env.jl")
include("lds.jl")
include("pendulum.jl")
include("utils.jl")
include("lqr.jl")


function lqr_control(env::ENV, U)
    X, U, c = rollout(env, U)
    costs = Float64[]
    push!(costs, c)
    @printf("(LQR) Initial cost %f\n", c)
    # linearize around trajectory
    ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u = linearize_quadraticize_traj(env, X, U)
    # Solve lqr
    k, K = lqr(env, ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u)    
    # Get new cost
    _, _, c = rollout(env, U, k=k, K=K, X_old=X)
    @printf("(lqr) Final cost %f\n", c)
    push!(costs, c)
    return costs
end

function model_based_control(env::ENV, model::ENV, U, T, alpha; line_search=false)
    _, _, c = rollout(env, U)
    costs = Float64[]
    push!(costs, c)

    X, U, cmodel = rollout(model, U)
    @printf("(Model-based) Initial cost %f\n", c)
    for t=1:T
        alphac = alpha
        # linearize around trajectory
        ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u = linearize_quadraticize_traj(model, X, U)
        # Solve lqr
        k, K = lqr(model, ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u)
        if line_search
            for i=1:20
                Xnew, Unew, cmodel_new = rollout(model, U, k=k, K=K, X_old=X, alpha=alphac)
                if cmodel_new < cmodel
                    @printf("Using a step length %f\n", alphac)
                    X, U, cmodel = copy(Xnew), copy(Unew), cmodel_new              
                    break
                end
                alphac = 0.5 * alphac
            end
        else
            # rollout
            X, U, cmodel = rollout(model, U, k=k, K=K, X_old=X, alpha=alphac)
        end
        # Get new cost
        _, _, c = rollout(env, U, k=k, K=K, X_old=X, alpha=alphac)
        @printf("(Model-based) cost %f\n", c)
        push!(costs, c)
    end
    @printf("(Model-based) Final cost %f\n", c)
    return costs
end

function ilc_loop(env::ENV, model::ENV, U, T, alpha; line_search=false)
    X, U, c = rollout(env, U)
    costs = Float64[]
    push!(costs, c)
    @printf("(ILC) Initial cost %f\n", c)

    for t=1:T
        alphac = alpha
        # linearize around trajectory
        ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u = linearize_quadraticize_traj(model, X, U)
        # Solve lqr
        k, K = lqr(model, ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u)
        if line_search
            alphac = alpha
            for i=1:20
                Xnew, Unew, cnew = rollout(env, U, k=k, K=K, X_old=X, alpha=alphac)
                if cnew < c
                    @printf("Using a step length %f\n", alphac)
                    X, U, c = copy(Xnew), copy(Unew), cnew
                    break
                end
                alphac = 0.5 * alphac                    
            end
        else
            # rollout
            X, U, c = rollout(env, U, k=k, K=K, X_old=X, alpha=alphac)
        end
        @printf("(ILC) cost %f\n", c)
        push!(costs, c)
    end
    @printf("(ILC) Final cost %f\n", c)
    return costs
end


function main_lds()
    state_size = 2
    action_size = 1

    # A = zeros(state_size, state_size)
    # B = zeros(state_size, action_size)
    # A = [1. 1.; -3. 1.]
    # B[1, 1] = 1.
    # B[2, 1] = 3.
    A = randn(state_size, state_size)
    B = randn(state_size, action_size)

    # eps_A = 2.
    # eps_B = 2.
    eps_A = randn()
    eps_B = randn()
    Ahat = A .+ eps_A * ones(state_size, state_size)
    Bhat = B .+ eps_B * ones(state_size, action_size)

    Q = Matrix{Float64}(I, state_size, state_size)
    R = Matrix{Float64}(I, action_size, action_size)

    H = 10
    T = 100
    alpha = 1.0

    x0 = 0.1 * ones(state_size)

    env = LDS(state_size, action_size, x0, A, B, Q, R, H)
    model = LDS(state_size, action_size, x0, Ahat, Bhat, Q, R, H)

    @printf("\n==============================================\n")
    # lqr control
    U_init = zeros(H, action_size)
    lqr_costs = lqr_control(env, U_init);
    min_cost = lqr_costs[end]

    @printf("\n==============================================\n")
    # Model-based control
    U_init = zeros(H, action_size)
    model_based_costs = model_based_control(env, model, U_init, 1, 1.0);
    model_based_cost = model_based_costs[end]
    normalized_model_based_cost = model_based_cost - min_cost
    
    @printf("\n==============================================\n")
    # ILC
    U_init = zeros(H, action_size)
    ilc_costs = ilc_loop(env, model, U_init, T, alpha, line_search=true);
    normalized_ilc_costs = [x - min_cost for x in ilc_costs]

    # Plot
    PyPlot.plot(1:T+1, normalized_ilc_costs, label="ILC")
    PyPlot.plot(1:T+1, [normalized_model_based_cost for i=1:T+1], label="Model-based")
    xlabel("Iterations")
    ylabel("Normalized cost")
    yscale("log")
    legend()

    @printf("Done")
end

function main_pendulum()
    H = 20
    x0 = [pi/2; 0.5]
    alpha = 1.0
    T = 100

    env = Pendulum(H, x0, 1.0)
    model = Pendulum(H, x0, 0.9)

    @printf("\n==============================================\n")
    # iLQR using true dynamics
    U_init = zeros(H, env.action_size)
    ilqr_costs = model_based_control(env, env, U_init, T, alpha, line_search=true)

    @printf("\n==============================================\n")
    # iLQR using model dynamics
    U_init = zeros(H, env.action_size)
    model_based_costs = model_based_control(env, model, U_init, T, alpha, line_search=true)

    @printf("\n==============================================\n")
    # ILC using model dynamics
    U_init = zeros(H, env.action_size)
    ilc_costs = ilc_loop(env, model, U_init, T, alpha, line_search=true)
    @printf("Done")

    PyPlot.plot(1:T+1, ilqr_costs, label="iLQR true")
    PyPlot.plot(1:T+1, model_based_costs, label="iLQR model")
    PyPlot.plot(1:T+1, ilc_costs, label="ILC")
    xlabel("Iterations")
    ylabel("Cost")
    legend()
end

end # module
