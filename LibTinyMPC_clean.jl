#ADMM types
# NSTATES = 12
# NINPUTS = 4
# NHORIZON = 10
# NTOTAL = 301

mutable struct TinyCache
    rho::Float64
    Kinf::Matrix{Float64}
    Pinf::Matrix{Float64}
    Quu_inv::Matrix{Float64}
    AmBKt::Matrix{Float64}
    APf::Vector{Float64}
    BPf::Vector{Float64}
end

mutable struct TinySettings
    abs_pri_tol::Float64;
    abs_dua_tol::Float64;
    max_iter::Int;
    check_termination::Int;
    en_state_bound::Int;  
    en_input_bound::Int;
    en_state_soc::Int;
    en_input_soc::Int;
end

mutable struct TinyBounds
    umin::Matrix{Float64}
    umax::Matrix{Float64}
    xmin::Matrix{Float64}
    xmax::Matrix{Float64}
    z::Matrix{Float64}
    znew::Matrix{Float64}
    v::Matrix{Float64}
    vnew::Matrix{Float64}
    y::Matrix{Float64}
    g::Matrix{Float64}
end

mutable struct TinySocs
    ncu::Int  # number of input cones (max 2)
    ncx::Int  # number of state cones (max 2)
    muu::Vector{Float64}  # coefficients for input cones
    mux::Vector{Float64}  # coefficients for state cones
    qcu::Vector{Int}  # dimensions for input cones (2 or 3)
    qcx::Vector{Int}  # dimensions for state cones (2 or 3)
    Acu::Vector{Int}  # start indexes for input cones
    Acx::Vector{Int}  # start indexes for state cones
    zc::Vector{Matrix{Float64}}  # input slack variables
    vc::Vector{Matrix{Float64}}  # state slack variables
    vcnew::Vector{Matrix{Float64}}  
    zcnew::Vector{Matrix{Float64}}
    gc::Vector{Matrix{Float64}}  # state dual variables
    yc::Vector{Matrix{Float64}}  # input dual variables
end

mutable struct TinyWorkspace
    x::Matrix{Float64}
    u::Matrix{Float64}
    q::Matrix{Float64}
    r::Matrix{Float64}
    p::Matrix{Float64}
    d::Matrix{Float64}

    pri_res_state::Float64
    pri_res_input::Float64
    dua_res_state::Float64
    dua_res_input::Float64
    status::Int
    iter::Int

    Q::Matrix{Float64}
    R::Matrix{Float64}
    Adyn::Matrix{Float64}
    Bdyn::Matrix{Float64}
    fdyn::Vector{Float64}
    
    Xref::Matrix{Float64}
    Uref::Matrix{Float64}
    Qu::Matrix{Float64}

    bounds::TinyBounds
    # socs::TinySocs
end

mutable struct TinySolver
    settings::TinySettings
    cache::TinyCache
    workspace::TinyWorkspace
end

# ADMM functions

# This one works like codegen
function backward_pass!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    work.R = work.R + cache.rho*I
    cache.Pinf = work.Q # put your terminal cost here
    for k = 500:-1:1  # enough iterations
        # print(cache.Pinf, "\n")
        cache.Kinf = (work.R + work.Bdyn'*cache.Pinf*work.Bdyn)\(work.Bdyn'*cache.Pinf*work.Adyn)
        cache.Pinf = work.Q + cache.Kinf'*work.R*cache.Kinf + (work.Adyn-work.Bdyn*cache.Kinf)'*cache.Pinf*(work.Adyn-work.Bdyn*cache.Kinf)
    end
    cache.AmBKt = (work.Adyn-work.Bdyn*cache.Kinf)'
    cache.APf = cache.AmBKt*cache.Pinf*work.fdyn
    cache.BPf = work.Bdyn'*cache.Pinf*work.fdyn
    cache.Quu_inv = (work.R + work.Bdyn'*cache.Pinf*work.Bdyn)\I
    work.p[:,NHORIZON] = -cache.Pinf*work.Xref[:,NHORIZON] # important
end 

# This is the actual backward pass computed online
function backward_pass_grad!(solver::TinySolver)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    work = solver.workspace
    cache = solver.cache
    for k = (NHORIZON-1):-1:1
        work.d[:,k] = cache.Quu_inv*(work.Bdyn'*work.p[:,k+1] + work.r[:,k] + cache.BPf)
        work.p[:,k] = work.q[:,k] + cache.AmBKt*work.p[:,k+1] - cache.Kinf'*work.r[:,k] + cache.APf
    end
end

function forward_pass!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    for k = 1:(NHORIZON-1)
        work.u[:,k] = -cache.Kinf*work.x[:,k] - work.d[:,k] 
        work.x[:,k+1] = work.Adyn*work.x[:,k] + work.Bdyn*work.u[:,k] + work.fdyn
    end
end

function update_primal!(solver::TinySolver)
    backward_pass_grad!(solver::TinySolver)
    forward_pass!(solver::TinySolver)
end

# function project_hyperplane(solver::TinySolver)
#     x_xyz = x[1:3]
#     if a'*x_xyz - b <= 0
#         return x
#     else
#         denom = 1 # Normalize in update loop
#         x_xyz_new = x_xyz - (a'*x_xyz - b)*a
#         return [x_xyz_new; x[4:end]]
#     end
# end

# function project_soc(x, mu)
#     n = 3 #length(x)
#     s = x[n] * mu
#     # display(x[1:3])
#     # print(x[1:3])
#     v = view(x,1:n-1)
#     a = norm(v)
#     if a <= -s  # below the cone
#         # print("below")
#         return [zeros(n); x[n+1:end]]
#     elseif a <= s  # in the code
#         return x
#     elseif a >= abs(s)  # outside the cone
#         # print("outside")
#         # print(size([0.5 * (1 + s/a) * [v; a/mu]; x[n+1:end]] ))
#         return [0.5 * (1 + s/a) * [v; a/mu]; x[n+1:end]] 
#     end
# end

function update_slack!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    stgs = solver.settings
    bounds = work.bounds

    umax = bounds.umax
    umin = bounds.umin
    xmax = bounds.xmax
    xmin = bounds.xmin

    #This function clamps the controls to be within the bounds
    for k = 1:(NHORIZON-1)
        bounds.z[:,k] = work.u[:,k]+bounds.y[:,k]
        bounds.v[:,k] = work.x[:,k]+bounds.g[:,k]

        if stgs.en_input_bound == 1
            bounds.z[:,k] .= min.(umax[:,k], max.(umin[:,k], bounds.z[:,k]))
        end

        if stgs.en_state_bound == 1
            bounds.v[:,k] .= min.(xmax[:,k], max.(xmin[:,k], bounds.v[:,k]))  # box
        end
        
        # if stgs.en_soc_state == 1
        #     v[k] .= project_soc(g[k] + x[k], stgs.mu)  # soc
        # end

        # if stgs.en_hplane_state == 1
        #     v[k] .= project_hyperplane(0, 0, g[k] + x[k], stgs.Acx[k], stgs.bcx[k])  # half-space 
        # end        
    end

    bounds.v[:,NHORIZON] = work.x[:,NHORIZON]+bounds.g[:,NHORIZON]
    if stgs.en_state_bound == 1
        bounds.v[:,NHORIZON] .= min.(xmax[:,NHORIZON], max.(xmin[:,NHORIZON], bounds.v[:,NHORIZON]))  # box
    end

    # if stgs.en_soc_state == 1
    #     v[NHORIZON] .= project_soc(x[NHORIZON]+g[NHORIZON], stgs.mu)  # soc
    # end

    # if stgs.en_hplane_state == 1
    #     v[NHORIZON] .= project_hyperplane(0, 0, g[NHORIZON] + x[NHORIZON], params.Acx[NHORIZON], params.bcx[NHORIZON])  
    # end
end

function update_dual!(solver::TinySolver)
    work = solver.workspace
    bounds = work.bounds
    #This function performs the standard AL multiplier update.
    #Note that we're using the "scaled form" where y = λ/ρ
    for k = 1:(NHORIZON-1)
        bounds.y[:,k] .= bounds.y[:,k] + work.u[:,k] - bounds.z[:,k]
        bounds.g[:,k] .= bounds.g[:,k] + work.x[:,k] - bounds.v[:,k]
    end
    bounds.g[:,NHORIZON] .= bounds.g[:,NHORIZON] + work.x[:,NHORIZON] - bounds.v[:,NHORIZON]
end

function update_linear_cost!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    bounds = work.bounds
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    for k = 1:(NHORIZON-1)
        work.r[:,k] = -work.R*work.Uref[:,k] # original R (doesn't matter)
        work.r[:,k] -= cache.rho*(bounds.z[:,k]-bounds.y[:,k])   
        work.q[:,k] = -work.Q*work.Xref[:,k]
        work.q[:,k] -= cache.rho*(bounds.v[:,k]-bounds.g[:,k]) 
    end
    work.p[:,NHORIZON] = -cache.Pinf*work.Xref[:,NHORIZON]
    work.p[:,NHORIZON] -= cache.rho*(bounds.v[:,NHORIZON]-bounds.g[:,NHORIZON])
end

#Main algorithm loop
function solve_admm!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    bounds = work.bounds
    stgs = solver.settings

    # forward_pass!(K,d,x,u,params,adaptive_step)
    # update_slack!(u,z,y,params)
    # update_dual!(u,z,y)
    update_linear_cost!(solver)

    work.pri_res_input = 1.0
    work.dua_res_input = 1.0
    work.pri_res_state = 1.0
    work.dua_res_state = 1.0
    work.status = 0
    work.iter = 0
    for k = 1:stgs.max_iter
        #Solver linear system with Riccati
        update_primal!(solver::TinySolver)

        #Project z into feasible domain
        update_slack!(solver::TinySolver)

        #Dual ascent
        update_dual!(solver::TinySolver)

        update_linear_cost!(solver::TinySolver)
        
        work.pri_res_input = maximum(abs.(work.u-bounds.znew))
        work.dua_res_input = maximum(abs.(cache.rho*(bounds.znew-bounds.z)))
        work.pri_res_state = maximum(abs.(work.x-bounds.vnew))
        work.dua_res_state = maximum(abs.(cache.rho*(bounds.vnew-bounds.v)))
                
        bounds.v .= bounds.vnew
        bounds.z .= bounds.znew

        work.iter += 1
        
        if (work.pri_res_input < stgs.abs_pri_tol && 
            work.dua_res_input < stgs.abs_dua_tol &&
            work.pri_res_state < stgs.abs_pri_tol && 
            work.dua_res_state < stgs.abs_dua_tol)
            work.status = 1
            break
        end
    end
    # display("Maximum iteration reached!")
    return bounds.znew[1], work.status, work.iter
end

function mat_from_vec(X::Vector{Vector{Float64}})::Matrix
    # convert a vector of vectors to a matrix 
    Xm = hcat(X...)
    return Xm 
end

function export_mat_to_c(declare, data)
    str = "sfloat" * declare * "= {\n"
    dataT = data'
    for i = 1:size(dataT, 1)
        str = str * "  "
        for j = 1:size(dataT, 2)
            this_str = @sprintf("%.6f", dataT[i, j])
            str = str * this_str * "f,"
        end
        str = str * "\n"
    end
    str = str * "};"
    return str
end
