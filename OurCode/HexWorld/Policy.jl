include("./HexWorld.jl")

# Init  policy evaluation : ulity 
# Symbol Upi(s) for state s and action pi(s)/a
function lookahead(𝒫::MDP, U::Vector{Float64}, s::Int64, a::Int64)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R[s, a] + γ * sum(T[s, a, s′] * U[s′] for s′ in 𝒮)
end

#Policy evaluation is done iteratively
#Symbol myPolicy with k_max iterations
function iterative_policy_evaluation(𝒫::MDP, myPolicy, k_max)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    U = [0.0 for s in 𝒮]
    for k in 1:k_max
        U = [lookahead(𝒫, U, s, myPolicy(s)) for s in 𝒮]
    end
    return U
end

#Value Function Policy
struct ValueFunctionPolicy
    𝒫::MDP # problem
    U::Vector{Float64} # utility function
end
#policy improvement
function greedy(𝒫::MDP, U, s)
    u, a = findmax(a -> lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a = a, u = u)
end
(myPolicy::ValueFunctionPolicy)(s) = greedy(myPolicy.𝒫, myPolicy.U, s).a


#Policy iteration => compute an optimal policy 
#It involves iterating between policy evaluation and policy improvement 
function solve(𝒫::MDP, myPolicy, k_max)
    𝒮 = 𝒫.𝒮
    for k = 1:k_max
        U = iterative_policy_evaluation(𝒫, myPolicy, k_max) #policy evaluation
        myPolicy′ = ValueFunctionPolicy(𝒫, U) #policy improvement
        if all(myPolicy(s) == myPolicy′(s) for s in 𝒮) #converge
            break
        end
        myPolicy = myPolicy′

    end
    return myPolicy
end


