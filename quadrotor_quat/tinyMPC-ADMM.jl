#ADMM Functions
function backward_pass!(A,B,Q,q,R,r,P,p,K,d,params)
    cache = params.cache
    N = params.N 
    P[N] .= cache.Pinf
    q[N] = -params.Qf * params.Xref[N]
    p[N] .= q[N]
    #This is the standard Riccati backward pass with both linear and quadratic terms (like iLQR)
    #Cache data and use IHLQR to save memory
    for k = (N-1):-1:1
        r[k] .= -params.R*params.Uref[k]
        K[k] .= (R + B'*P[k+1]*B)\(B'*P[k+1]*A)
        d[k] .= (R + B'*P[k+1]*B)\(B'*p[k+1] + r[k])
        
        q[k] .= -params.Q * params.Xref[k]
        P[k] .= Q + K[k]'*R*K[k] + (A-B*K[k])'*P[k+1]*(A-B*K[k])
        p[k] .= q[k] + (A-B*K[k])'*(p[k+1]-P[k+1]*B*d[k]) + K[k]'*(R*d[k]-r[k])
    end
    # cache.Kinf .= K[1]
    # cache.Pinf .= P[1]
    # cache.Quu_inv .= (R + B'*P[1]*B)\I
    # cache.AmBKt .= (A-B*K[1])'
end

function backward_pass_grad!(A,B,q,R,r,P,p,K,d,params)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    N = params.N 
    cache = params.cache
    for k = (N-1):-1:1
        d[k] .= cache.Quu_inv*(B'*p[k+1] + r[k])
        p[k] .= q[k] + cache.AmBKt*(p[k+1] - cache.Pinf*B*d[k]) + cache.Kinf'*(R*d[k] - r[k])
    end
end

function forward_pass!(A,B,K,d,x,u,params)
    N = params.N 
    for k = 1:(N-1)
        u[k] .= -K[k]*x[k] - d[k] 
        x[k+1] .= A*x[k] + B*u[k]
    end
end

function update_primal!(A,B,q,R̃,r,P,p,K,d,x,u,params)
    backward_pass_grad!(A,B,q,R̃,r,P,p,K,d,params)
    forward_pass!(A,B,K,d,x,u,params)
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

function update_linear_cost!(z,y,r,ρ,params)
    N = params.N
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    for k = 1:(N-1)
        r[k] .= -ρ*(z[k]-y[k]) - params.R*params.Uref[k]
    end    
end

#Main algorithm loop
function solve_admm!(params,A,B,q,R̃,r,P,p,K,d,x,u,z,znew,y;ρ=1.0,abs_tol=1e-2,max_iter=200)
    forward_pass!(A,B,K,d,x,u,params)
    update_slack!(u,z,y,params)
    update_dual!(u,z,y)
    update_linear_cost!(z,y,r,ρ,params)

    primal_residual = 1.0
    dual_residual = 1.0
    for iter = 1:max_iter
        #Solver linear system with Riccati
        update_primal!(A,B,q,R̃,r,P,p,K,d,x,u,params)

        #Project z into feasible domain
        update_slack!(u,znew,y,params)

        #Dual ascent
        update_dual!(u,znew,y)

        update_linear_cost!(znew,y,r,ρ,params)
        
        primal_residual = maximum(abs.(mat_from_vec(u)-mat_from_vec(znew)))
        dual_residual = maximum(abs.(ρ*(mat_from_vec(znew)-mat_from_vec(z))))
        
        z .= znew
        
        if (primal_residual > abs_tol || dual_residual > abs_tol)
            if (verbose == 1) 
                display("Success!")
            end
            break
        end
    end
    # display("Maximum iteration reached!")
    return z[1]
end

function mat_from_vec(X::Vector{Vector{Float64}})::Matrix
    # convert a vector of vectors to a matrix 
    Xm = hcat(X...)
    return Xm 
end