using GeometryBasics: HyperSphere

#ADMM Functions
function backward_pass_grad!(q, r, p, d, params)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    cache = params.cache
    ρ_k = params.ρ_index[1]
    for k = (params.N-1):-1:1
        d[k] .= cache.Quu_inv_list[ρ_k]*(cache.B'*p[k+1] + r[k])
        p[k] .= q[k] + cache.AmBKt_list[ρ_k]*p[k+1] - cache.Kinf_list[ρ_k]'*r[k] + cache.coeff_d2p_list[ρ_k]*d[k]
    end
end

function forward_pass!(d, x, u, params)
    cache = params.cache
    ρ_k = params.ρ_index[1]
    for k = 1:(params.N-1)
        u[k] .= -cache.Kinf_list[ρ_k]*x[k] - d[k] 
        x[k+1] .= cache.A*x[k] + cache.B*u[k]
    end
end

# Updates x and u (decision variables)
function update_primal!(q, r, p, d, x, u, params)
    backward_pass_grad!(q, r, p, d, params)
    forward_pass!(d, x, u, params)
end

function project_hyperplane(k, vis, x, A, b)
    a = A[1:3]
    x_xyz = x[1:3]
    if a'*x_xyz - b <= 0
        return x
    else
        denom = 1 # Normalize in update loop
        x_xyz_new = x_xyz - (a'*x_xyz - b)*a

        if k == -1
            delete!(vis["z_location"])
            delete!(vis["z_proj_location"])
            setobject!(vis["z_location"], HyperSphere(Point{3, Float64}(x_xyz[1], x_xyz[2], 0.5), .02),
                LineBasicMaterial(color=Colors.RGBA(0,0,1)))
            setobject!(vis["z_proj_location"], HyperSphere(Point{3, Float64}(x_xyz_new[1], x_xyz_new[2], 0.5), .02),
                LineBasicMaterial(color=Colors.RGBA(1,0,0)))
        end

        return [x_xyz_new; x[4:end]]
    end
end

function update_slack!(vis, x, v, g, u, z, y, params)
    #This function clamps the controls to be within the bounds
    for k = 1:(params.N-1)
        z[k] .= min.(params.umax, max.(params.umin, u[k] + y[k]))
    end
    for k = 1:params.N
        v[k] .= project_hyperplane(k, vis, x[k] + g[k], params.constraint_A[k], params.xmax[k][1])
    end
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

function update_linear_cost!(v, g, z, y, p, q, r, params)
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    ρ_k = params.ρ_index[1]
    for k = 1:(params.N-1)
        r[k] .= -params.cache.ρ_list[ρ_k][1]*(z[k] - y[k]) - params.R*params.Uref[k] # original R
        q[k] .= -params.cache.ρ_list[ρ_k][1]*(v[k] - g[k]) - params.Q*params.Xref[k] 
    end
    p[params.N] .= -params.cache.ρ_list[ρ_k][1]*(v[params.N] - g[params.N]) - params.Qf*params.Xref[params.N]
end

#Main algorithm loop
function solve_admm!(vis, params, q, r, p, d, x,v,vnew,g, u,z,znew,y; abs_tol=1e-2, max_iter=200, iters_check_rho_update=1)

    primal_residual = 1.0
    primal_residual_state = 1.0
    dual_residual_input = 1.0
    dual_residual_state = 1.0
    status = 0
    iter = 1

    forward_pass!(d, x, u, params)
    update_slack!(vis, x, v, g, u, z, y, params)
    update_dual!(x, v, g, u, z, y, params)
    update_linear_cost!(v, g, z, y, p, q, r, params)
    for k = 1:max_iter
        #Solver linear system with Riccati
        update_primal!(q, r, p, d, x, u, params)

        #Project z into feasible domain
        update_slack!(vis, x, vnew, g, u, znew, y, params)

        #Dual ascent
        update_dual!(x, vnew, g, u, znew, y, params)

        update_linear_cost!(vnew, g, znew, y, p, q, r, params)
        
        primal_residual = maximum(abs.(hcat(u...) - hcat(znew...)))
        primal_residual_state = maximum(abs.(hcat(x...) - hcat(vnew...)))
        dual_residual_input = maximum(abs.(params.cache.ρ_list[params.ρ_index[1]][1]*(hcat(znew...) - hcat(z...))))
        dual_residual_state = maximum(abs.(params.cache.ρ_list[params.ρ_index[1]][1]*(hcat(vnew...) - hcat(v...))))

        
        z = deepcopy(znew)
        v = deepcopy(vnew)

        
        if (primal_residual < abs_tol && 
            primal_residual_state < abs_tol &&
            dual_residual_input < abs_tol &&
            dual_residual_state < abs_tol)
            status = 1
            break
        end

        if k % iters_check_rho_update == 0
            if dual_residual_input > abs_tol/100
                ρ_scale = sqrt( (max(primal_residual, primal_residual_state)/max(norm(u, Inf), norm(znew, Inf))) / (dual_residual_input/max(norm(znew, Inf), norm(z, Inf))) )
                if ρ_scale >= 5
                    ρ_increase = floor(Int, log(5, ρ_scale))
                    ρ_choose = params.ρ_index[1]
                    ρ_choose = max(params.ρ_index[1], min(ρ_choose + ρ_increase, length(params.cache.ρ_list)))
                    # if ρ_choose != params.ρ_index[1]
                    #     display("updating ρ from " * string(params.cache.ρ_list[params.ρ_index[1]][1][1]) * " to " * string(params.cache.ρ_list[ρ_choose][1][1]))
                    # end
                    params.ρ_index .= ρ_choose
                end
            end
        end

        iter += 1

    end

    return z, status, iter
end