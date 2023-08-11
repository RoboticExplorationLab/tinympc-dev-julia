using GeometryBasics: HyperSphere

#ADMM Functions
function backward_pass_grad!(q, r, p, d, params)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    cache = params.cache
    for k = (params.N-1):-1:1
        # display("short")
        d[k] .= cache.Quu_inv*(cache.B̃'*p[k+1] + r[k])
        p[k] .= q[k] + cache.AmBKt*p[k+1] - cache.Kinf'*r[k] + cache.coeff_d2p*d[k]
        # p[k] .= q[k] + cache.Ã'*cache.Pinf*cache.B̃*d[k] - cache.Ã'*p[k+1]
        # display(q[k])
        # display(p[k])
    end
end

function forward_pass!(d, x, u, params)
    cache = params.cache
    for k = 1:(params.N-1)
        u[k] .= -cache.Kinf*x[k] - d[k] 
        x[k+1] .= cache.Ã*x[k] + cache.B̃*u[k]
    end
end

# Updates x and u (decision variables)
function update_primal!(q, r, p, d, x, u, params)
    backward_pass_grad!(q, r, p, d, params)
    forward_pass!(d, x, u, params)
end

function project_hyperplane(k, vis, x, A, q)
    a = A[1:2]
    x_xy = x[1:2]
    if a'*x_xy - a'*q <= 0
        return x
    else
        x_xy_new = [a[2]^2 -a[1]*a[2]; -a[1]*a[2] a[1]^2]/(a[1]^2+a[2]^2) * (x_xy-q) + q

        if k == 1
            # display(a'*x_xy_new - a'*q)

            # sleep(.1)
            # Visualize solution
            delete!(vis["z_location"])
            delete!(vis["z_proj_location"])
            setobject!(vis["z_location"], HyperSphere(Point{3, Float64}(x_xy[1], x_xy[2], .5), .02),
                LineBasicMaterial(color=Colors.RGBA(0,0,1)))
            setobject!(vis["z_proj_location"], HyperSphere(Point{3, Float64}(x_xy_new[1], x_xy_new[2], .5), .02),
                LineBasicMaterial(color=Colors.RGBA(1,0,0)))
        end

        return [x_xy_new; x[3:end]]
    end
end

function update_slack!(vis, x, v, g, u, z, y, params)
    #This function clamps the controls to be within the bounds
    for k = 1:(params.N-1)
        z[k] .= min.(params.umax, max.(params.umin, u[k] + y[k])) # TODO: convert u[k] + y[k] to params.A*u[k] + params.b + y[k] or combine into one slack and dual update
    end
    for k = 1:params.N
        # v[k] .= min.(params.xmax[k], max.(params.xmin[k], x[k] + g[k]))
        v[k] .= project_hyperplane(k, vis, x[k] + g[k], params.A[k], params.xmax[k])
    end
    # display(xmax[1][1])
end

function update_dual!(x, v, g, u, z, y, params)
    #This function performs the standard AL multiplier update.
    #Note that we're using the "scaled form" where y = λ/ρ
    for k = 1:(params.N-1)
        y[k] .= y[k] + u[k] - z[k]
    end
    for k = 1:params.N
        g[k] .= g[k] + x[k] - v[k]
    end
end

function update_linear_cost!(v, g, z, y, p, q, r, ρ, params)
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    for k = 1:(params.N-1)
        r[k] .= -ρ*(z[k]-y[k]) - params.R*params.Uref[k] # original R
        q[k] .=  -params.Q*(params.Xref[k] - g[k])
    end
    p[params.N] .= -params.cache.Pinf*(params.Xref[params.N] - g[params.N])
end

#Main algorithm loop
function solve_admm!(vis, params, q, r, p, d, x,v,vnew,g, u,z,znew,y; ρ=1.0, abs_tol=1e-2, max_iter=200)

    primal_residual = 1.0
    dual_residual_input = 1.0
    dual_residual_state = 1.0
    status = 0
    iter = 1

    forward_pass!(d, x, u, params)
    update_slack!(vis, x, v, g, u, z, y, params)
    update_dual!(x, v, g, u, z, y, params)
    update_linear_cost!(v, g, z, y, p, q, r, ρ, params)
    for k = 1:max_iter
    # for k = 1:1
        #Solver linear system with Riccati
        update_primal!(q, r, p, d, x, u, params)

        #Project z into feasible domain
        update_slack!(vis, x, vnew, g, u, znew, y, params)

        #Dual ascent
        update_dual!(x, vnew, g, u, znew, y, params)

        update_linear_cost!(vnew, g, znew, y, p, q, r, ρ, params)
        
        primal_residual = maximum(abs.(hcat(u...) - hcat(znew...)))
        dual_residual_input = maximum(abs.(ρ*(hcat(znew...) - hcat(z...))))
        # dual_residual_state = maximum(abs.(ρ*(hcat(vnew...) - hcat(v...))))

        
        z = deepcopy(znew)
        v = deepcopy(vnew)

        
        if (primal_residual < abs_tol && 
            dual_residual_input < abs_tol) # &&
            # dual_residual_state < abs_tol)
            status = 1
            break
        end

        iter += 1


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
