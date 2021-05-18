module ILC

using Plots
using PyPlot
using ForwardDiff
using LinearAlgebra
using Printf
using Parameters

import RobotZoo.Acrobot
using StaticArrays
using SparseArrays
using TrajectoryOptimization
import TrajectoryOptimization: get_times
using RobotDynamics
using Random
using Colors
using TrajOptPlots
using Altro
using MeshCat
using OSQP

include("env.jl")
include("lds.jl")
include("pendulum.jl")
# include("acrobot.jl")
include("utils.jl")
include("lqr.jl")
include("qp.jl")

function lqr_control(env::ENV, U; verbose=false)
    X, U, c = rollout(env, U)
    costs = Float64[]
    push!(costs, c)
    @printf("(LQR) Initial cost %f\n", c)
    # linearize around trajectory
    ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u = linearize_quadraticize_traj(env,
                                                                       X,
                                                                       U)
    # Solve lqr
    k, K = lqr(env, ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u)
    # Get new cost
    Xnew, Unew, c = rollout(env, U, k=k, K=K, X_old=X)
    @printf("(lqr) Final cost %f\n", c)
    push!(costs, c)
    return Xnew, Unew, costs
end

function ricatti_solution(x0, A, B, Ahat, Bhat, Q, R, Qf, H)

    # Compute tvlqr solution
    K, P = tvlqr([Ahat for i=1:H], [Bhat for i=1:H], Q, R, Qf)

    # Compute cost and trajectory
    X, U, c = rollout(x0, A, B, Q, R, Qf, H, K)
    return X, U, c
end

function model_based_control(env::ENV, model::ENV, U, T, alpha;
                             line_search=false, verbose=false)
    _, _, c = rollout(env, U)
    costs = Float64[]
    push!(costs, c)

    Xmodel, Umodel, cmodel = rollout(model, U)
    @printf("(Model-based) Initial cost %f\n", c)
    for t=1:T
        alphac = alpha
        # linearize around trajectory
        ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u = linearize_quadraticize_traj(
            model, Xmodel, Umodel)
        # Solve lqr
        k, K = lqr(model, ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u)
        if line_search
            alpha_found = false
            for i=1:20
                Xmodel_new, Umodel_new, cmodel_new = rollout(
                    model, Umodel, k=k, K=K, X_old=Xmodel, alpha=alphac)
                if cmodel_new < cmodel
                    if verbose
                        @printf("Using a step length %f\n", alphac)
                    end
                    Xmodel_updated, Umodel_updated, cmodel = copy(Xmodel_new),
                    copy(Umodel_new), cmodel_new
                    alpha_found = true
                    break
                end
                alphac = 0.5 * alphac
            end
            if !alpha_found
                Xmodel_updated, Umodel_updated, cmodel = copy(Xmodel),
                copy(Umodel), cmodel
            end
        else
            # rollout
            Xmodel_updated, Umodel_updated, cmodel = rollout(model, Umodel, k=k,
                                                             K=K, X_old=Xmodel,
                                                             alpha=alphac)
        end
        # Get new cost
        _, _, c = rollout(env, Umodel, k=k, K=K, X_old=Xmodel, alpha=alphac)
        Umodel = deepcopy(Umodel_updated)
        Xmodel = deepcopy(Xmodel_updated)
        if verbose
            @printf("(Model-based) cost %f\n", c)
        end
        push!(costs, c)
    end
    @printf("(Model-based) Final cost %f\n", c)
    return Xmodel, Umodel, costs
end

function ilc_loop(env::ENV, model::ENV, U, T, alpha; line_search=false,
                  verbose=false)
    X, U, c = rollout(env, U)
    costs = Float64[]
    push!(costs, c)
    @printf("(ILC) Initial cost %f\n", c)

    for t=1:T
        alphac = alpha
        # linearize around trajectory
        ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u = linearize_quadraticize_traj(
            model, X, U)
        # Solve lqr
        k, K = lqr(model, ∇F_x, ∇F_u, ∇C_x, ∇C_u, ∇2C_x, ∇2C_u)
        if line_search
            alphac = alpha
            for i=1:100
                Xnew, Unew, cnew = rollout(env, U, k=k, K=K, X_old=X,
                                           alpha=alphac)
                if cnew < c
                    if verbose
                        @printf("Using a step length %f\n", alphac)
                    end
                    X, U, c = copy(Xnew), copy(Unew), cnew
                    break
                end
                alphac = 0.9 * alphac
            end
        else
            # rollout
            X, U, c = rollout(env, U, k=k, K=K, X_old=X, alpha=alphac)
        end
        if verbose
            @printf("(ILC) cost %f\n", c)
        end
        push!(costs, c)
    end
    @printf("(ILC) Final cost %f after %d iterations\n", c, T)
    return X, U, costs
end

function ricatti_ilc_solution(x0, A, B, Ahat, Bhat, Q, R, Qf, H)

    # Compute tvilc solution
    K, P = tvilc([A for i=1:H], [B for i=1:H], [Ahat for i=1:H],
                 [Bhat for i=1:H], Q, R, Qf)
    # Compute cost and trajectory
    X, U, c = rollout(x0, A, B, Q, R, Qf, H, K)
    return X, U, c

end

function compute_sse(Xsim, Xref)
    N = length(Xref)
    sse = 0.
    for k=1:N
         sse = sse + norm(Xsim[k] - Xref[k])^2
    end
    return sse
end

function run_ilc(model::Acrobot, env::Acrobot, altro, goal_constraint;
        verbose=true,
        tol=1e-2
    )
    # Get nominal trajectory
    Xref = states(altro)
    Uref = controls(altro);
    times = get_times(altro)
    n,m,Nh = size(altro)

    # Get costs out of solver
    Q = get_objective(altro).obj[1].Q
    R = get_objective(altro).obj[1].R
    Qf = get_objective(altro).obj[end].Q

    # Get bound out of solver
    u_bnd = get_constraints(altro).convals[1].con.z_max[end]

    # ILC Cost functions
    Qilc = Diagonal(SA[1.0; 1.0; 1.0; 1.0])
    Rilc = Diagonal(SA[0.1])

    # Generate ILC Data
    data = ILCData(Qilc,Rilc,Qf, Xref, Uref, get_times(altro), u_bnd);

    A = [zeros(n,n) for k = 1:Nh-1]
    B = [zeros(n,m) for k = 1:Nh-1]
    ∇f = RobotDynamics.DynamicsJacobian(model)
    for k = 1:Nh-1
        local t = times[k]
        local dt = times[k+1] - times[k]
        z = KnotPoint(SVector{n}(Xref[k]), SVector{m}(Uref[k]), dt, t)
        discrete_jacobian!(RK4, ∇f, model, z)
        A[k] .= ∇f.A
        B[k] .= ∇f.B
    end

    K, P = tvlqr(A, B, Q, R, Qf);
    U = kron(I(Nh-1), [I zeros(m,n)])

    Xsim, Usim = rollout(env, K, Xref, Uref, times)

    qp = build_qp(data, Xsim, Usim, A,B,K)

    goal_constraint_violations = []
    push!(goal_constraint_violations, norm(Xsim[end] - goal_constraint.xf))
    sse_old = compute_sse(Xsim, Xref)

    while true
        res = OSQP.solve!(qp)
        z = res.x
        Δu = U * z
        for k=1:Nh-1
            sidx = (k-1) * (m)
            Uref[k] = Uref[k] + Δu[sidx+1:sidx+m]
        end
        Xsim, Usim = rollout(env, K, Xref, Uref, times)
        update_qp!(qp, data, Xsim, Usim)
        sse_new = compute_sse(Xsim, Xref)
        push!(goal_constraint_violations, norm(Xsim[end] - goal_constraint.xf))
        verbose && @printf(" sse: %f, goal_constraint_violation: %f\n", sse_new,
                           norm(Xsim[end] - goal_constraint.xf))
        if abs(sse_new - sse_old) < tol
            break
        end
        sse_old = sse_new
    end

    return Xsim, Usim, goal_constraint_violations
end

function exp_lds()

    # epsilons = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0]
    epsilons = []
    ϵ = 1e-3
    while ϵ <= 10.0
        push!(epsilons, ϵ)
        ϵ *= 1.5
    end
    print(epsilons)
    model_based_costs = []
    ilc_costs = []
    for i=1:length(epsilons)
        ce_cost, ilc_cost = main_lds(epsilons[i], true)
        push!(model_based_costs, ce_cost)
        push!(ilc_costs, ilc_cost)
    end
    PyPlot.plot(epsilons, model_based_costs, label="CE")
    PyPlot.plot(epsilons, ilc_costs, label="ILC")
    #PyPlot.plot([3.1622, 3.1622], [minimum(ilc_costs), maximum(model_based_costs)])
    xlabel("ϵ")
    ylabel("Cost Suboptimality gap")
    yscale("log")
    xscale("log")
    legend()
    title("Linear Dynamical System with approximate model")
    println(model_based_costs)
    println(ilc_costs)
end

function main_lds(eps, no_plot=false)
    dimension = 1
    state_size = 2 * dimension
    action_size = dimension
    h = 0.5  # stepsize

    A = zeros(state_size, state_size)
    B = zeros(state_size, action_size)
    A = [1. 1.; -3. 1.]
    B[1, 1] = 1.
    B[2, 1] = 3.
    # A = randn(state_size, state_size)
    # B = randn(state_size, action_size)

    # Double integrator dynamics
    # for i=1:dimension
    #     A[i, dimension+i] = 1.
    #     B[dimension+i, i] = 1.
    # end
    # A[1, 1] = 1.0
    # A[1, 2] = h
    # A[2, 1] = 0.0
    # A[2, 2] = 1.0
    # B[1, 1] = 0.5*h^2
    # B[2, 1] = h

    println(opnorm(A))
    println(opnorm(B))

    eps_A = eps
    eps_B = eps
    # eps_A = randn()
    # eps_B = randn()
    # Ahat = A .+ eps_A * ones(state_size, state_size)
    # Bhat = B .+ eps_B * ones(state_size, action_size)
    Ahat = A .+ eps_A * Matrix{Float64}(I, state_size, state_size)
    Bhat = B .+ eps_B * Matrix{Float64}(I, state_size, action_size)

    @printf("eps_A is %f\n", opnorm(Ahat-A))
    @printf("eps_B is %f\n", opnorm(Bhat-B))

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
    _, _, init_cost = rollout(env, U_init)
    # Xlqr, Ulqr, lqr_costs = lqr_control(env, U_init);
    # min_cost = lqr_costs[end]
    Xlqr, Ulqr, lqr_costs = model_based_control(env, env, U_init, T, 1.0)
    min_cost = lqr_costs[end]

    Xricatti, Uricatti, ricatti_costs = ricatti_solution(x0, A, B, A, B, Q, R, Q, H)

    # println(Ulqr)
    # println(lqr_costs[end])
    # println(Uricatti)
    # println(ricatti_costs)

    @printf("\n==============================================\n")
    # Model-based control
    U_init = zeros(H, action_size)
    Xmodel, Umodel, model_based_costs = model_based_control(env, model, U_init, T, 1.0);
    model_based_cost = model_based_costs[end]
    normalized_model_based_cost = model_based_cost - min_cost
    normalized_model_based_costs = [init_cost - min_cost]
    for k=1:T
        push!(normalized_model_based_costs, normalized_model_based_cost)
    end

    Xmodelricatti, Umodelricatti, modelricatti_costs = ricatti_solution(
        x0, A, B, Ahat, Bhat, Q, R, Q, H)

    # println(Umodel)
    # println(model_based_costs[end])
    # println(Umodelricatti)
    # println(modelricatti_costs)
    
    @printf("\n==============================================\n")
    # ILC
    U_init = zeros(H, action_size)
    Xilc, Uilc, ilc_costs = ilc_loop(env, model, U_init, T, alpha, line_search=true);
    normalized_ilc_costs = [x - min_cost for x in ilc_costs]

    Xilcricatti, Uilcricatti, ilcricatti_costs = ricatti_ilc_solution(x0, A, B,
                                                                      Ahat,
                                                                      Bhat, Q,
                                                                      R, Q, H)
    # println(Uilc)
    # println(ilc_costs[end])
    # println(Uilcricatti)
    # println(ilcricatti_costs)

    # Plot
    # if !no_plot
    #     PyPlot.plot(0:T, normalized_ilc_costs, label="ILC")
    #     PyPlot.plot(0:T, normalized_model_based_costs,
    #                 label="CE")
    #     xlabel("Iterations")
    #     ylabel("Normalized cost")
    #     yscale("log")
    #     legend()
    # end

    @printf("Done\n")
    # return normalized_model_based_costs[end], normalized_ilc_costs[end]
    return modelricatti_costs - ricatti_costs, ilcricatti_costs - ricatti_costs
end

function exp_pendulum()

    # Δms = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    Δms = []
    Δm = 0.0
    while Δm <= 0.25
        push!(Δms, Δm)
        Δm += 0.01
    end
    ilc_costs = []
    ce_costs = []
    for Δm in Δms
        ce_cost, ilc_cost = main_pendulum(Δm, true)
        push!(ce_costs, ce_cost)
        push!(ilc_costs, ilc_cost)
    end

    PyPlot.plot(Δms, ce_costs, label="CE")
    PyPlot.plot(Δms, ilc_costs, label="ILC")
    PyPlot.plot(Δms, [60.022 for _ in Δms], label="Initial Controls")
    xlabel("Δm")
    ylabel("Cost suboptimality gap")
    legend()
    title("Inverted Pendulum with misspecified mass")
end

function main_pendulum(Δm, no_plot=false)
    H = 20
    x0 = [pi/2; 0.5]
    alpha = 0.1
    T = 200

    env = Pendulum(H, x0, 1.0)
    model = Pendulum(H, x0, 1.0 + Δm)

    @printf("\n==============================================\n")
    # iLQR using true dynamics
    U_init = zeros(H, env.action_size)
    _, _, init_cost = rollout(env, U_init)
    @printf("initial cost %f\n", init_cost)
    _, _, ilqr_costs = model_based_control(env, env, U_init, T, alpha,
                                     line_search=true)

    @printf("\n==============================================\n")
    # iLQR using model dynamics
    U_init = zeros(H, env.action_size)
    _, _, model_based_costs = model_based_control(env, model, U_init, T, alpha,
                                            line_search=true)

    @printf("\n==============================================\n")
    # ILC using model dynamics
    U_init = zeros(H, env.action_size)
    _, _, ilc_costs = ilc_loop(env, model, U_init, T, alpha, line_search=true)
    @printf("Done")

    if !no_plot
        PyPlot.plot(1:T+1, ilqr_costs, label="iLQR true")
        PyPlot.plot(1:T+1, model_based_costs, label="iLQR model")
        PyPlot.plot(1:T+1, ilc_costs, label="ILC")
        xlabel("Iterations")
        ylabel("Cost")
        legend()
    end

    return model_based_costs[end] - ilqr_costs[end], ilc_costs[end] - ilqr_costs[end]
end

function main_acrobot()
    l_model = @SVector [1.0, 1.0]
    m_model = @SVector [1.0, 1.0]
    J_model = @SVector [(1.0/12)*m_model[1]*l_model[1]*l_model[1],
                        (1.0/12)*m_model[2]*l_model[2]*l_model[2]]

    l_true = @SVector [1.0-0.005, 1.0+0.004]
    m_true = @SVector [1.0+0.01, 1.0-0.01]
    J_true = @SVector [(1.0/12)*m_true[1]*l_true[1]*l_true[1],
                        (1.0/12)*m_true[2]*l_true[2]*l_true[2]]

    model = Acrobot(l_model, m_model, J_model, false)
    env = Acrobot(l_true, m_true, J_true, true)

    n, m = size(model)

    # Discretization
    N = 101
    Tf = 5.0
    h = Tf/(N-1)

    # Initial and Final Conditions
    x0 = @SVector [-pi/2, 0, 0, 0]
    xf = @SVector [pi/2, 0, 0, 0];  # i.e. swing up

    # Objective
    Q = 100.0*Diagonal(@SVector ones(n))
    Qf = 1000.0*Diagonal(@SVector ones(n))
    R = 1.0*Diagonal(@SVector ones(m))
    obj = LQRObjective(Q,R,Qf,xf,N);

    # Constraints
    conSet = ConstraintList(n,m,N)

    #   Control bounds
    u_bnd = 20.0
    bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    add_constraint!(conSet, bnd, 1:N-1)

    #   Goal constraint
    goal = GoalConstraint(xf)
    add_constraint!(conSet, goal, N)

    # Define the problem
    prob_model = Problem(model, obj, xf, Tf, x0=x0, constraints=conSet,
                         integration=RK4);
    prob_true = Problem(env, obj, xf, Tf, x0=x0, constraints=conSet,
                        integration=RK4);


    # Solve problem using Altro
    opts = SolverOptions(
        iterations=100000,
        constraint_tolerance=1e-4,
        projected_newton=0,
        verbose=0
    )

    Random.seed!(1)
    U0 = [randn() for k = 1:N-1]
    initial_controls!(prob_model, U0)
    initial_controls!(prob_true, U0)

    altro_model = ALTROSolver(prob_model, opts)
    altro_true = ALTROSolver(prob_true, opts)
    set_options!(altro_model, show_summary=true)
    set_options!(altro_true, show_summary=true)
    solve!(altro_model);
    solve!(altro_true)

    # Visualizer
    vis = Visualizer()
    TrajOptPlots.set_mesh!(vis, model)
    # render(vis)
    # visualize!(vis, model, Tf, states(altro))

    # Extract reference trajectory
    Xref_model = states(altro_model)
    Uref_model = controls(altro_model)
    times_model = get_times(altro_model)
    Xref_true = states(altro_true)
    Uref_true = controls(altro_true)
    times_true = get_times(altro_true)

    # Linearize around reference trajectory
    A_model, B_model = linearize_around_ref(model, Xref_model, Uref_model,
                                            times_model)

    # Compute certainty equivalent tracking controller
    K_model, P_model = tvlqr(A_model, B_model, Q, R, Qf)

    # Get trajectory by certainty equivalent control
    X_model, U_model = rollout(env, K_model, states(altro_model),
                               controls(altro_model), get_times(altro_model))

    # Linearize true dynamics around reference trajectory
    A_true, B_true = linearize_around_ref(env, Xref_true, Uref_true, times_true)

    # Compute tracking controller using true dynamics
    K_true, P_true = tvlqr(A_true, B_true, Q, R, Qf)

    # Get trajectory by true control
    X_true, U_true = rollout(env, K_true, states(altro_true),
                             controls(altro_true),
                             get_times(altro_true))

    # Evaluate terminal constraint violation
    model_constraint_violation = norm(goal.xf - X_model[end])
    true_constraint_violation = norm(goal.xf - X_true[end])
    @printf("Certainty equivalent control violates goal constraint by %f\n",
            model_constraint_violation)
    @printf("True control violates goal constraint by %f\n",
            true_constraint_violation)

    # Run ILC
    X_ilc, U_ilc, ilc_constraint_violations = run_ilc(model, env, altro_model,
                                                      goal, tol=5e-3)

    # Visualize
    render(vis)
    visualize!(vis, model, Tf, X_model, X_true, X_ilc, colors=[colorant"blue",
                                                               colorant"red",
                                                               colorant"green"])

    # Compare goal constraint violations
    T = length(ilc_constraint_violations)
    PyPlot.plot(0:T-1, ilc_constraint_violations, label="ilc")
    PyPlot.plot(0:T-1, [model_constraint_violation for i=0:T-1], label="certainty
    equivalent")
    PyPlot.plot(0:T-1, [true_constraint_violation for i=0:T-1], label="true dynamics")
    xlabel("Iterations")
    ylabel("Goal Constraint Violation")
    legend();
end

end # module
