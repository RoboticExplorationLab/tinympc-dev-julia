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

function update_linear_cost!(z, y, p, q, r, ρ, params)
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    for k = 1:(params.N-1)
        r[k] .= -ρ*(z[k]-y[k]) - params.R*params.Uref[k]  # original R
        q[k] .= -params.Q*params.Xref[k]
    end
    p[N] .= -P[N]*params.Xref[N]
end

#Main algorithm loop
function solve_admm!(params, q, r, p, d, x, u, z, znew, y; ρ=1.0, abs_tol=1e-2, max_iter=200)
    # forward_pass!(K,d,x,u,params,adaptive_step)
    # update_slack!(u,z,y,params)
    # update_dual!(u,z,y)
    # update_linear_cost!(z,y,p,q,r,ρ,params)

    # println("q:\n", q)
    # println("r:\n", r)
    # println("p:\n", p)
    # println("d:\n", d)
    # println("q:\n", q)
    # println("u:\n", u)
    # println("z:\n", z)
    # println("znew:\n", znew)
    # println("y:\n", y)

    primal_residual = 1.0
    dual_residual = 1.0
    status = 0
    iter = 0
    # for k = 1:max_iter
    for k = 1:5
        #Solver linear system with Riccati
        update_primal!(q, r, p, d, x, u, params)
        
        if k == 5
            println("k = 2\n")
            println("q:\n", q)
            println("r:\n", r)
            println("p:\n", p)
            println("d:\n", d)
        end


        #Project z into feasible domain
        update_slack!(u, znew, y, params)

        #Dual ascent
        update_dual!(u, znew, y, params)

        update_linear_cost!(znew, y, p, q, r, ρ, params)
        
        primal_residual = maximum(abs.(mat_from_vec(u)-mat_from_vec(znew)))
        dual_residual = maximum(abs.(ρ*(mat_from_vec(znew)-mat_from_vec(z))))
        
        z .= znew

        iter += 1
        
        if (primal_residual < abs_tol && dual_residual < abs_tol)
            status = 1
            break
        end
    end
    # display("Maximum iteration reached!")
    return znew[1], status
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
