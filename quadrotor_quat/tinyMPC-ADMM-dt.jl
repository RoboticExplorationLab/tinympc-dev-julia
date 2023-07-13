#ADMM Functions
function backward_pass_grad!(q, r, p, d, params)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    cache = params.cache
    for k = (params.N-1):-1:1
        # display("short")
        d[k] .= cache.Quu_inv*(cache.B̃'*p[k+1] + r[k])
        p[k] .= q[k] + cache.AmBKt*p[k+1] - cache.Kinf'*r[k] + cache.coeff_d2p*d[k]
    end
end

function forward_pass!(d, x, u, params)
    cache = params.cache
    for k = 1:(params.N-1)
        u[k] .= -cache.Kinf*x[k] - d[k] 
        x[k+1] .= cache.Ã*x[k] + cache.B̃*u[k]
    end
end

function update_primal!(q, r, p, d, x, u, params)
    backward_pass_grad!(q, r, p, d, params)
    forward_pass!(d, x, u, params)
end

function update_slack!(u, z, y, params)
    #This function clamps the controls to be within the bounds
    for k = 1:(params.N-1)
        z[k] .= min.(params.umax, max.(params.umin, u[k]+y[k]))
    end
end

function update_dual!(u, z, y, params)
    #This function performs the standard AL multiplier update.
    #Note that we're using the "scaled form" where y = λ/ρ
    for k = 1:(params.N-1)
        y[k] .= y[k] + u[k] - z[k]
    end
end

function update_linear_cost!(x, z, y, p, q, r, ρ, params)
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    xref = params.Xref
    for k = 1:(params.N-1)
        r[k] .= -ρ*(z[k]-y[k]) - params.R*params.Uref[k]  # original R
        ϕ = qtorp( L(params.Xref[k][4:7])' * rptoq(x[k][4:6]) )
        Δx̃ = [x[k][1:3] - xref[k][1:3]; ϕ; x[k][7:9]-xref[k][8:10]; x[k][10:12]-xref[k][11:13]]
        q[k] .= -params.Q*Δx̃
    end
    ϕ_end = qtorp( L(params.Xref[N][4:7])' * rptoq(x[N][4:6]) )
    Δx̃_end = [x[N][1:3] - xref[N][1:3]; ϕ_end; x[N][7:9]-xref[N][8:10]; x[N][10:12]-xref[N][11:13]]
    p[N] .= -P[N]*Δx̃_end
    # p[N] .= -params.cache.Pinf * params.Xref[N]
end

#Main algorithm loop
function solve_admm!(params, q, r, p, d, x, u, z, znew, y; ρ=1.0, abs_tol=1e-2, max_iter=200)
    # forward_pass!(K,d,x,u,params,adaptive_step)
    # update_slack!(u,z,y,params)
    # update_dual!(u,z,y)
    # update_linear_cost!(z,y,p,q,r,ρ,params)


    primal_residual = 1.0
    dual_residual = 1.0
    status = 0
    iter = 0
    for k = 1:max_iter
    # for k = 1:5
        #Solver linear system with Riccati
        update_primal!(q, r, p, d, x, u, params)

        #Project z into feasible domain
        update_slack!(u, znew, y, params)

        #Dual ascent
        update_dual!(u, znew, y, params)

        update_linear_cost!(x, znew, y, p, q, r, ρ, params)

        # display(mat_from_vec(u))
        # display(mat_from_vec(znew))
        # display(mat_from_vec(z))
        
        primal_residual = maximum(abs.(mat_from_vec(u)-mat_from_vec(znew)))
        dual_residual = maximum(abs.(ρ*(mat_from_vec(znew)-mat_from_vec(z))))
        
        z .= znew

        iter += 1
        
        if (primal_residual < abs_tol && dual_residual < abs_tol)
            status = 1
            break
        end
    end

    # display(iter)
    # display(primal_residual)
    # display(dual_residual)

    # display("vals at end")
    # display(q)
    # display(r)
    # display(p)
    # display(d)
    # display(x)
    # display(u)
    # display(z)
    # display(y)
    # display(primal_residual)
    # display(dual_residual)

    return znew[1], status
    # return znew, status
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
