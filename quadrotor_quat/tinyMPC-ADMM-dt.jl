#ADMM Functions
function backward_pass!(Q,q,R,r,P,p,K,d,params,adaptive_step)
    cache = params.cache
    N = params.N 

    if adaptive_step > 0
        P[N] .= cache.Pinf2
        A = 1*Ã
        B = 1*B̃
        cache.Kinf .= cache.Quu_inv2*(B'*P[N]*A)
        cache.Pinf .= Q + cache.Kinf'*R*cache.Kinf + (A-B*cache.Kinf)'*P[N]*(A-B*cache.Kinf)
        cache.Quu_inv .= (R + B'*cache.Pinf*B)\I
        cache.AmBKt .= (A-B*cache.Kinf)'
        cache.coeff_d2p .= cache.Kinf'*R - cache.AmBKt*cache.Pinf*B
    else 
        P[N] .= cache.Pinf
    end    
end

function backward_pass_grad!(q,R,r,P,p,K,d,params,adaptive_step)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    N = params.N 
    cache = params.cache
    for k = (N-1):-1:1
        if (adaptive_step > 0 && k > adaptive_step)
            # display("long")
            d[k] .= cache.Quu_inv2*(B̃s'*p[k+1] + r[k])
            p[k] .= q[k] + cache.AmBKt2*p[k+1] - cache.Kinf2'*r[k] + cache.coeff_d2p2*d[k]
        else
            # display("short")
            d[k] .= cache.Quu_inv*(B̃'*p[k+1] + r[k])
            p[k] .= q[k] + cache.AmBKt*p[k+1] - cache.Kinf'*r[k] + cache.coeff_d2p*d[k]
        end
    end
end

function forward_pass!(K,d,x,u,params,adaptive_step)
    N = params.N 
    for k = 1:(N-1)
        if (adaptive_step > 0 && k > adaptive_step)
            A = 1*Ãs
            B = 1*B̃s
            u[k] .= -cache.Kinf2*x[k] - d[k] 
            x[k+1] .= A*x[k] + B*u[k]
        else
            A = 1*Ã
            B = 1*B̃
            u[k] .= -cache.Kinf*x[k] - d[k] 
            x[k+1] .= A*x[k] + B*u[k]
        end
    end
end

function update_primal!(q,R,r,P,p,K,d,x,u,params,adaptive_step)
    backward_pass_grad!(q,R,r,P,p,K,d,params,adaptive_step)
    forward_pass!(K,d,x,u,params,adaptive_step)
end

function update_slack!(u,z,y,params)
    N = params.N 
    umax = params.u_max
    umin = params.u_min
    #This function clamps the controls to be within the bounds
    for k = 1:(N-1)
        z[k] .= min.(umax, max.(umin, u[k]+y[k]))
    end
end

function update_dual!(u,z,y)
    N = params.N
    #This function performs the standard AL multiplier update.
    #Note that we're using the "scaled form" where y = λ/ρ
    for k = 1:(N-1)
        y[k] .= y[k] + u[k] - z[k]
    end
end

function update_linear_cost!(z,y,p,q,r,ρ,params)
    N = params.N
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    for k = 1:(N-1)
        r[k] .= -ρ*(z[k]-y[k]) - params.R*params.Uref[k]
        q[k] .= -params.Q*params.Xref[k]
    end    
    p[N] .= -P[N]*params.Xref[N]
end

#Main algorithm loop
function solve_admm!(params,q,R̃,r,P,p,K,d,x,u,z,znew,y;ρ=1.0,abs_tol=1e-2,max_iter=200,adaptive_step)
    # forward_pass!(K,d,x,u,params,adaptive_step)
    # update_slack!(u,z,y,params)
    # update_dual!(u,z,y)
    # update_linear_cost!(z,y,p,q,r,ρ,params)

    primal_residual = 1.0
    dual_residual = 1.0
    status = 0
    iter = 0
    for k = 1:max_iter
        #Solver linear system with Riccati
        update_primal!(q,R̃,r,P,p,K,d,x,u,params,adaptive_step)

        #Project z into feasible domain
        update_slack!(u,znew,y,params)

        #Dual ascent
        update_dual!(u,znew,y)

        update_linear_cost!(znew,y,p,q,r,ρ,params)
        
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
    return znew[1], status, iter
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
