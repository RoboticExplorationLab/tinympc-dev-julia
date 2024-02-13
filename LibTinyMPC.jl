#ADMM Functions
function backward_pass!(Q,R,P,params,adaptive_step)
    cache = params.cache
    N = params.N 

    if adaptive_step > 0
        P[N] .= cache.Pinf2
        A = 1*Ã
        B = 1*B̃
        cache.Quu_inv .= (R + B'*P[N]*B)\I0
        cache.Kinf .=  cache.Quu_inv*(B'*P[N]*A)
        cache.Pinf .= Q + cache.Kinf'*R*cache.Kinf + (A-B*cache.Kinf)'*P[N]*(A-B*cache.Kinf)        
        cache.AmBKt .= (A-B*cache.Kinf)'
    else 
        P[N] .= cache.Pinf
    end    
end

function backward_pass_grad!(q,r,p,d,params,adaptive_step)
    #This is just the linear/gradient term from the backward pass (no cost-to-go Hessian or K calculations)
    N = params.N 
    cache = params.cache
    for k = (N-1):-1:1
        if (adaptive_step > 0 && k > adaptive_step)
            # display("long")
            d[k] .= cache.Quu_inv2*(B̃s'*p[k+1] + r[k])
            p[k] .= q[k] + cache.AmBKt2*p[k+1] - cache.Kinf2'*r[k]
        else
            # display("short")
            d[k] .= cache.Quu_inv*(B̃'*p[k+1] + r[k])
            p[k] .= q[k] + cache.AmBKt*p[k+1] - cache.Kinf'*r[k]
        end
    end
end

function backward_pass_grad_aff!(A,B,f,Q,q,R,r,P,p,K,d)
    #This is the standard Riccati backward pass with both linear and quadratic terms (like iLQR)
    for k = (N-1):-1:1
        q[:,k] .= -Q*xref[:,k]
        K[:,:,k] .= (R + B'*P[:,:,k+1]*B)\(B'*P[:,:,k+1]*A)
        d[:,k] .= (R + B'*P[:,:,k+1]*B)\(B'*p[:,k+1] + B'*P[:,:,k+1]*f + r[:,k])
        r[:,k] .= -R*uref[:,k]
        P[:,:,k] .= Q + 1*K[:,:,k]'*R*K[:,:,k] + (A-1*B*K[:,:,k])'*P[:,:,k+1]*(A-B*K[:,:,k])
        # p[:,k] .= q[:,k] + (A-B*K[:,:,k])'*(p[:,k+1]-P[:,:,k+1]*B*d[:,k]) + K[:,:,k]'*(R*d[:,k]-r[:,k])
        p[:,k] .= q[:,k] + (A-B*K[:,:,k])'*p[:,k+1] - K[:,:,k]'*r[:,k] + (A-B*K[:,:,k])'*P[:,:,k+1]*f
    end
end

function forward_pass!(d,x,u,params,adaptive_step)
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

function update_primal!(q,r,p,d,x,u,params,adaptive_step)
    backward_pass_grad!(q,r,p,d,params,adaptive_step)
    forward_pass!(d,x,u,params,adaptive_step)
end

function project_hyperplane(k, vis, x, a, b)
    x_xyz = x[1:3]
    if a'*x_xyz - b <= 0
        return x
    else
        denom = 1 # Normalize in update loop
        x_xyz_new = x_xyz - (a'*x_xyz - b)*a
        return [x_xyz_new; x[4:end]]
    end
end

function project_soc(x, mu)
    n = 3 #length(x) -- size of each cone qc[k]
    s = x[n] * mu  # ||v|| = ||u[1:n-1]|| <= u[n] * mu = s
    v = view(x,1:n-1)
    a = norm(v)
    if a <= -s  # below the cone
        return [zeros(n); x[n+1:end]]
    elseif a <= s  # in the code
        return x
    elseif a >= abs(s)  # outside the cone
        return [0.5 * (1 + s/a) * [v; a/mu]; x[n+1:end]] 
    end
end

function update_slack!(x,zx,yx,u,zu,yu,params)
    N = params.N 
    umax = params.u_max
    umin = params.u_min
    #This function clamps the controls to be within the bounds
    for k = 1:(N-1)
        zu[k] = u[k]+yu[k]
        zx[k] = x[k]+yx[k]
        if params.en_box_input == 1
         zu[k] .= min.(umax, max.(umin, u[k]+yu[k]))
        end

        if params.en_box_state == 1
            zx[k] .= min.(xmax, max.(xmin, x[k]+yx[k]))  # box
        end
        
        if params.en_soc_state == 1
            zx[k] .= project_soc(yx[k] + x[k], params.mu)  # soc
        end

        if params.en_hplane_state == 1
            zx[k] .= project_hyperplane(0, 0, yx[k] + x[k], params.Acx[k], params.bcx[k])  # half-space 
        end        
    end

    zx[N] = x[N]+yx[N]
    if params.en_box_state == 1
        zx[N] .= min.(xmax, max.(xmin, x[N]+yx[N]))  # box
    end

    if params.en_soc_state == 1
        zx[N] .= project_soc(x[N]+yx[N], params.mu)  # soc
    end

    if params.en_hplane_state == 1
        zx[N] .= project_hyperplane(0, 0, yx[N] + x[N], params.Acx[N], params.bcx[N])  
    end

end

function update_dual!(x,zx,yx,u,zu,yu)
    N = params.N
    #This function performs the standard AL multiplier update.
    #Note that we're using the "scaled form" where y = λ/ρ
    for k = 1:(N-1)
        yu[k] .= yu[k] + u[k] - zu[k]
        yx[k] .= yx[k] + x[k] - zx[k]
    end
    yx[N] .= yx[N] + x[N] - zx[N]
end

function update_linear_cost!(zx,yx,zu,yu,p,q,r,ρ,params)
    N = params.N
    #This function updates the linear term in the control cost to handle the changing cost term from ADMM
    for k = 1:(N-1)
        r[k] .= -ρ*(zu[k]-yu[k]) - params.R*params.Uref[k]  # original R
        q[k] .= -ρ*(zx[k]-yx[k]) - params.Q*params.Xref[k]
    end    
    p[N] .= -ρ*(zx[N]-yx[N]) - P[N]*params.Xref[N]
end

#Main algorithm loop
function solve_admm!(params,q,r,p,d,x,zx,zx_new,yx,u,zu,zu_new,yu;ρ=1.0,abs_tol=1e-2,max_iter=200,adaptive_step)
    # forward_pass!(K,d,x,u,params,adaptive_step)
    # update_slack!(u,z,y,params)
    # update_dual!(u,z,y)
    # update_linear_cost!(z,y,p,q,r,ρ,params)

    pri_res_u = 1.0
    dua_res_u = 1.0
    pri_res_x = 1.0
    dua_res_x = 1.0
    status = 0
    iter = 0
    for k = 1:max_iter
        #Solver linear system with Riccati
        update_primal!(q,r,p,d,x,u,params,adaptive_step)

        #Project z into feasible domain
        update_slack!(x,zx_new,yx,u,zu_new,yu,params)

        #Dual ascent
        update_dual!(x,zx_new,yx,u,zu_new,yu)

        update_linear_cost!(zx_new,yx,zu_new,yu,p,q,r,ρ,params)
        
        pri_res_u = maximum(abs.(mat_from_vec(u)-mat_from_vec(zu_new)))
        dua_res_u = maximum(abs.(ρ*(mat_from_vec(zu_new)-mat_from_vec(zu))))
        pri_res_x = maximum(abs.(mat_from_vec(x)-mat_from_vec(zx_new)))
        dua_res_x = maximum(abs.(ρ*(mat_from_vec(zx_new)-mat_from_vec(zx))))
                
        zx .= zx_new
        zu .= zu_new

        iter += 1
        
        if (pri_res_u < abs_tol && 
            dua_res_u < abs_tol &&
            pri_res_x < abs_tol && 
            dua_res_x < abs_tol)
            status = 1
            break
        end
    end
    # display("Maximum iteration reached!")
    return zu_new[1], status, iter
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
