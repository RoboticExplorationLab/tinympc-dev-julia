using Printf

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
    ncu::Int  # number of input cones 
    ncx::Int  # number of state cones 
    cu::Vector{Float64}  # coefficients for input cones
    cx::Vector{Float64}  # coefficients for state cones
    qcu::Vector{Int}  # dimensions for input cones 
    qcx::Vector{Int}  # dimensions for state cones 
    Acu::Vector{Int}  # start indexes for input cones
    Acx::Vector{Int}  # start indexes for state cones
    zc::Vector{Matrix{Float64}}  # input slack variables
    zcnew::Vector{Matrix{Float64}}
    vc::Vector{Matrix{Float64}}  # state slack variables
    vcnew::Vector{Matrix{Float64}}  
    yc::Vector{Matrix{Float64}}  # input dual variables
    gc::Vector{Matrix{Float64}}  # state dual variables
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
    socs::TinySocs
end

mutable struct TinySolver
    settings::TinySettings
    cache::TinyCache
    workspace::TinyWorkspace
end

# ADMM functions

# This one works like codegen
function compute_cache!(solver::TinySolver, Q, R)
    work = solver.workspace
    cache = solver.cache
    work.R = R + cache.rho*I
    work.Q = Q + cache.rho*I
    cache.Pinf .= work.Q # put your terminal cost here
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
        # display(cache.Quu_inv)
        # display(work.Bdyn)
        # display(work.p[:,k+1])
        # display(work.r[:,k])
        # display(cache.BPf)
        work.d[:,k] = cache.Quu_inv*(work.Bdyn'*work.p[:,k+1] + work.r[:,k] + cache.BPf)
        work.p[:,k] = work.q[:,k] + cache.AmBKt*work.p[:,k+1] - cache.Kinf'*work.r[:,k] + cache.APf
    end
end

function forward_pass!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    for k = 1:(NHORIZON-1)
        # display(cache.Kinf)
        # display(work.d[:,k])
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

function project_soc(s::Vector{Float64}, mu::Float64, n::Int)
    """
    Project a vector `s` onto the second-order cone defined by `mu` and `n`
    s is already selected with Ac
    """
    u0 = s[n]*mu
    u1 = view(s,1:n-1)
    a = norm(u1)
    # display(s)
    # display(u0)
    # display(u1)
    # display(a)
    if a <= -u0  # below the cone
        # print("below")
        return zeros(n)
    elseif a <= u0  # in the code
        return (s)
    elseif a >= abs(u0)  # outside the cone
        # print("outside")
        return (0.5 * (1 + u0/a) * [u1; a/mu])
    end
end

function update_slack!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    stgs = solver.settings
    bounds = work.bounds
    socs = work.socs

    umax = bounds.umax
    umin = bounds.umin
    xmax = bounds.xmax
    xmin = bounds.xmin

    #This function clamps the controls to be within the bounds
    for k = 1:(NHORIZON-1)
        # compute the updated slack
        bounds.znew[:,k] = work.u[:,k] + bounds.y[:,k]
        bounds.vnew[:,k] = work.x[:,k] + bounds.g[:,k]
        for cone_i = 1:socs.ncu 
            socs.zcnew[cone_i][:,k]  = work.u[:,k] + socs.yc[cone_i][:,k]
        end
        for cone_i = 1:socs.ncx
            socs.vcnew[cone_i][:,k]  = work.x[:,k] + socs.gc[cone_i][:,k]
        end

        # project the updated slack
        if stgs.en_input_bound == 1
            bounds.znew[:,k] .= min.(umax[:,k], max.(umin[:,k], bounds.znew[:,k]))
        end

        if stgs.en_state_bound == 1
            bounds.vnew[:,k] .= min.(xmax[:,k], max.(xmin[:,k], bounds.vnew[:,k]))
            # display(bounds.vnew[:,k])
        end
        
        if stgs.en_input_soc == 1 && socs.ncu > 0
            for cone_i = 1:socs.ncu
                start = socs.Acu[cone_i]
                indexes = start:(start+socs.qcu[cone_i]-1)
                socs.zcnew[cone_i][indexes, k] .= project_soc(socs.zcnew[cone_i][indexes, k], socs.cu[cone_i], socs.qcu[cone_i])  # soc
            end
        end

        if stgs.en_state_soc == 1 && socs.ncx > 0
            for cone_i = 1:socs.ncx
                start = socs.Acx[cone_i]
                indexes = start:(start+socs.qcx[cone_i]-1)
                socs.vcnew[cone_i][indexes, k] .= project_soc(socs.vcnew[cone_i][indexes, k], socs.cx[cone_i], socs.qcx[cone_i])  # soc
            end
        end

        # if stgs.en_hplane_state == 1
        #     v[k] .= project_hyperplane(0, 0, g[k] + x[k], stgs.Acx[k], stgs.bcx[k])  # half-space 
        # end        
    end

    # update the last step slack
    bounds.vnew[:,NHORIZON] = work.x[:,NHORIZON] + bounds.g[:,NHORIZON]
    if stgs.en_state_bound == 1
        bounds.vnew[:,NHORIZON] .= min.(xmax[:,NHORIZON], max.(xmin[:,NHORIZON], bounds.vnew[:,NHORIZON]))  # box
    end

    if stgs.en_state_soc == 1 && socs.ncx > 0
        for cone_i = 1:socs.ncx
            socs.vcnew[cone_i][:,NHORIZON] = work.x[:,NHORIZON] + socs.gc[cone_i][:,NHORIZON]
            start = socs.Acx[cone_i]
            indexes = start:(start+socs.qcx[cone_i]-1)
            socs.vcnew[cone_i][indexes, NHORIZON] .= project_soc(socs.vcnew[cone_i][indexes, NHORIZON], socs.cx[cone_i], socs.qcx[cone_i])  # soc
        end
    end

    # if stgs.en_hplane_state == 1
    #     v[NHORIZON] .= project_hyperplane(0, 0, g[NHORIZON] + x[NHORIZON], params.Acx[NHORIZON], params.bcx[NHORIZON])  
    # end
end

function update_dual!(solver::TinySolver)
    work = solver.workspace
    bounds = work.bounds
    socs = work.socs
    #This function performs the standard AL multiplier update.
    #Note that we're using the "scaled form" where y = λ/ρ
    for k = 1:(NHORIZON-1)
        bounds.y[:,k] .= bounds.y[:,k] + work.u[:,k] - bounds.znew[:,k]
        bounds.g[:,k] .= bounds.g[:,k] + work.x[:,k] - bounds.vnew[:,k]
        if en_input_soc == 1
            for cone_i = 1:socs.ncu
                socs.yc[cone_i][:,k] .= socs.yc[cone_i][:,k] + work.u[:,k] - socs.zcnew[cone_i][:,k]
            end
        end
        if en_state_soc == 1
            for cone_i = 1:socs.ncx
                socs.gc[cone_i][:,k] .= socs.gc[cone_i][:,k] + work.x[:,k] - socs.vcnew[cone_i][:,k]
            end
        end
    end

    bounds.g[:,NHORIZON] .= bounds.g[:,NHORIZON] + work.x[:,NHORIZON] - bounds.vnew[:,NHORIZON]
    if en_state_soc == 1
        for cone_i = 1:socs.ncx
            socs.gc[cone_i][:,NHORIZON]  = work.x[:,NHORIZON] + socs.gc[cone_i][:,NHORIZON]
        end
    end
end

function update_linear_cost!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    bounds = work.bounds
    socs = work.socs
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    for k = 1:(NHORIZON-1)
        work.r[:,k] = -(work.R-cache.rho*I)*work.Uref[:,k] # original R??
        work.r[:,k] -= cache.rho*(bounds.znew[:,k] - bounds.y[:,k])  
        if en_input_soc == 1                
            for cone_i = 1:socs.ncu
                work.r[:,k] -= cache.rho*(socs.zcnew[cone_i][:,k] - socs.yc[cone_i][:,k])
            end 
        end
        work.q[:,k] = -(work.Q-cache.rho*I)*work.Xref[:,k]
        work.q[:,k] -= cache.rho*(bounds.vnew[:,k] - bounds.g[:,k])
        # display(norm(work.q[:,k]))
        if en_state_soc == 1
            for cone_i = 1:socs.ncx
                work.q[:,k] -= cache.rho*(socs.vcnew[cone_i][:,k] - socs.gc[cone_i][:,k])
            end
        end
    end
    work.p[:,NHORIZON] = -cache.Pinf*work.Xref[:,NHORIZON]
    work.p[:,NHORIZON] -= cache.rho*(bounds.vnew[:,NHORIZON] - bounds.g[:,NHORIZON])
    if en_state_soc == 1
        for cone_i = 1:socs.ncx
            work.p[:,NHORIZON] -= cache.rho*(socs.vcnew[cone_i][:,NHORIZON] - socs.gc[cone_i][:,NHORIZON])
        end
    end
end

function reset_dual!(solver)
    work = solver.workspace
    bounds = work.bounds
    socs = work.socs
    bounds.y .= zeros(NINPUTS, NHORIZON-1)
    bounds.g .= zeros(NSTATES, NHORIZON)
    if en_input_soc == 1
        for cone_i = 1:socs.ncu
            socs.yc[cone_i] .= zeros(NINPUTS, NHORIZON-1)
        end
    end
    if en_state_soc == 1
        for cone_i = 1:socs.ncx
            socs.gc[cone_i] .= zeros(NSTATES, NHORIZON)
        end
    end

end

#Main algorithm loop
function solve_admm!(solver::TinySolver)
    work = solver.workspace
    cache = solver.cache
    bounds = work.bounds
    stgs = solver.settings
    socs = work.socs

    # reset_dual!(solver)
    # forward_pass!(solver)
    # update_slack!(solver)
    # update_dual!(solver)
    # update_linear_cost!(solver)

    # bounds.v .= bounds.vnew
    # bounds.z .= bounds.znew
    # socs.vc .= socs.vcnew
    # socs.zc .= socs.zcnew

    work.pri_res_input = 1.0
    work.dua_res_input = 1.0
    work.pri_res_state = 1.0
    work.dua_res_state = 1.0
    work.status = 0
    work.iter = 0
    for k = 1:stgs.max_iter
        #Solver linear system with Riccati
        update_primal!(solver)

        # display(work.x)
        # display(work.u)

        #Project z into feasible domain
        update_slack!(solver)

        #Dual ascent
        update_dual!(solver)

        update_linear_cost!(solver)
        
        work.pri_res_input = maximum(abs.(work.u-bounds.znew))
        work.dua_res_input = maximum(abs.(cache.rho*(bounds.znew-bounds.z)))

        if en_state_soc == 1 && socs.ncu > 0
            for cone_i = 1:socs.ncu
                work.pri_res_input = max(work.pri_res_input, maximum(abs.(work.u-socs.zcnew[cone_i])))
                work.dua_res_input = max(work.dua_res_input, maximum(abs.(cache.rho*(socs.zcnew[cone_i]-socs.zc[cone_i]))))
            end
        end

        work.pri_res_state = maximum(abs.(work.x-bounds.vnew))
        work.dua_res_state = maximum(abs.(cache.rho*(bounds.vnew-bounds.v)))

        if en_input_soc == 1 && socs.ncx > 0
            for cone_i = 1:socs.ncx
                work.pri_res_state = max(work.pri_res_state, maximum(abs.(work.x-socs.vcnew[cone_i])))
                work.dua_res_state = max(work.dua_res_state, maximum(abs.(cache.rho*(socs.vcnew[cone_i]-socs.vc[cone_i]))))
            end
        end
        
        bounds.v .= bounds.vnew
        bounds.z .= bounds.znew
        socs.vc .= socs.vcnew
        socs.zc .= socs.zcnew

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
    str = "tinytype " * declare * " = {\n"
    for i = 1:size(data, 1)
        str = str * "\t"
        for j = 1:size(data, 2)
            this_str = @sprintf("%.6f", data[i, j])
            if i == size(data,1) && j == size(data,2)
                str = str * this_str * "f"
            else
                str = str * this_str * "f, "
            end
        end
        str = str * "\n"
    end
    str = str * "};"
    return str
end

function export_vec_to_c(declare, data)
    str = "tinytype " * declare * " = {"
    for i = 1:size(data,1)
        this_str = @sprintf("%.6f", data[i])
        if i == size(data,1)
            str = str * this_str * "f"
        else
            str = str * this_str * "f, "
        end
    end
    str = str * "};"
    return str
end


function export_diag_to_c(declare, data)
    str = "tinytype " * declare * " = {"
    for i = 1:size(data,1)
        this_str = @sprintf("%.6f", data[i, i])
        if i == size(data,1)
            str = str * this_str * "f"
        else
            str = str * this_str * "f, "
        end
    end
    str = str * "};"
    return str
end


function reset_solver!(solver)
    solver.cache.Kinf = zeros(NINPUTS, NSTATES)
    solver.cache.Pinf = zeros(NSTATES, NSTATES)
    solver.cache.Quu_inv = zeros(NINPUTS, NINPUTS)
    solver.cache.AmBKt = zeros(NSTATES, NSTATES)
    solver.cache.APf = zeros(NSTATES)
    solver.cache.BPf = zeros(NINPUTS)
    solver.workspace.bounds.z = zeros(NINPUTS, NHORIZON-1)
    solver.workspace.bounds.znew = zeros(NINPUTS, NHORIZON-1)
    solver.workspace.bounds.v = zeros(NSTATES, NHORIZON)
    solver.workspace.bounds.vnew = zeros(NSTATES, NHORIZON)
    solver.workspace.bounds.y = zeros(NINPUTS, NHORIZON-1)
    solver.workspace.bounds.g = zeros(NSTATES, NHORIZON)
    solver.workspace.socs.zc = [zeros(NINPUTS, NHORIZON-1) for i = 1:2]
    solver.workspace.socs.zcnew = [zeros(NINPUTS, NHORIZON-1) for i = 1:2]
    solver.workspace.socs.vc = [zeros(NSTATES, NHORIZON) for i = 1:2]
    solver.workspace.socs.vcnew = [zeros(NSTATES, NHORIZON) for i = 1:2]
    solver.workspace.socs.yc = [zeros(NINPUTS, NHORIZON-1) for i = 1:2]
    solver.workspace.socs.gc = [zeros(NSTATES, NHORIZON) for i = 1:2]
    solver.workspace.x = zeros(NSTATES, NHORIZON)
    solver.workspace.u = zeros(NINPUTS, NHORIZON-1)
    solver.workspace.q = zeros(NSTATES, NHORIZON)
    solver.workspace.r = zeros(NINPUTS, NHORIZON-1)
    solver.workspace.p = zeros(NSTATES, NHORIZON)
    solver.workspace.d = zeros(NINPUTS, NHORIZON-1)
    solver.workspace.pri_res_state = 1.0
    solver.workspace.pri_res_input = 1.0
    solver.workspace.dua_res_state = 1.0
    solver.workspace.dua_res_input = 1.0
    solver.workspace.status = 0
    solver.workspace.iter = 0
end