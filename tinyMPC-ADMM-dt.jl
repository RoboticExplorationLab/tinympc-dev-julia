#ADMM Functions
function backward_pass_grad!(q, r, p, d, params, N)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    cache = params.cache
    for k = (N-1):-1:1
        # display("short")
        d[k] .= cache.Quu_inv*(cache.B̃'*p[k+1] + r[k])
        p[k] .= q[k] + cache.AmBKt*p[k+1] - cache.Kinf'*r[k] + cache.coeff_d2p*d[k]
    end
end

function forward_pass!(d, x, u, params, N)
    cache = params.cache
    for k = 1:(N-1)
        u[k] .= -cache.Kinf*x[k] - d[k] 
        x[k+1] .= cache.Ã*x[k] + cache.B̃*u[k]
    end
end

function update_primal!(q, r, p, d, x, u, params, N)
    backward_pass_grad!(q, r, p, d, params, N)
    forward_pass!(d, x, u, params, N)
end

function update_slack!(u, z, y, params, N)
    #This function clamps the controls to be within the bounds
    for k = 1:(N-1)
        z[k] .= min.(params.umax, max.(params.umin, u[k]+y[k]))
    end
end

function update_dual!(u, z, y, params, N)
    #This function performs the standard AL multiplier update.
    #Note that we're using the "scaled form" where y = λ/ρ
    for k = 1:(N-1)
        y[k] .= y[k] + u[k] - z[k]
    end
end

function update_linear_cost!(x, z, y, p, q, r, ρ, params, N)
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    xref = params.Xref
    for k = 1:(N-1)
        r[k] .= -ρ*(z[k]-y[k]) - params.R*params.Uref[k]  # original R
        q[k] .= -params.Q*params.Xref[k]
    end
    p[N] .= -params.cache.Pinf*params.Xref[N]
end

#Main algorithm loop
function solve_admm!(vis, params, q, r, p, d, x, u, z, znew, y; ρ=1.0, abs_tol=1e-2, max_iter=200)

    primal_residual = 1.0
    dual_residual = 1.0
    status = 0
    iter = 1

    N = length(x) # maybe check that length(x) = len(p,q) = len(d,r,y,znew,z,u)-1

    forward_pass!(d, x, u, params, N)
    update_slack!(u, z, y, params, N)
    update_dual!(u, z, y, params, N)
    update_linear_cost!(x, z, y, p, q, r, ρ, params, N)
    for k = 1:max_iter
        #Solver linear system with Riccati
        update_primal!(q, r, p, d, x, u, params, N)

        #Project z into feasible domain
        update_slack!(u, znew, y, params, N)

        #Dual ascent
        update_dual!(u, znew, y, params, N)

        update_linear_cost!(x, znew, y, p, q, r, ρ, params, N)
        
        primal_residual = maximum(abs.(hcat(u...) - hcat(znew...)))
        dual_residual = maximum(abs.(ρ*(hcat(znew...) - hcat(z...))))
        
        z = deepcopy(znew)

        
        if (primal_residual < abs_tol && dual_residual < abs_tol)
            status = 1
            break
        end

        iter += 1

        # sleep(.5)
        
        # Visualize solution
        # delete!(vis["xhistLine"])
        # xhistLine = [Point(x_[1], x_[2], x_[3]) for x_ in x]
        # setobject!(vis["xhistLine"], Object(PointCloud(xhistLine), 
        # LineBasicMaterial(color=Colors.RGBA(1,0.6,0.)), "Line"))

    end

    return u, status, iter
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
