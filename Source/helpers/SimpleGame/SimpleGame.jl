include("Helper.jl")
struct SimpleGame
    γ # discount factor
    ℐ # agents
    𝒜 # joint action space
    R # joint reward function
end

# Policy is a action-probability dictionary 
struct SimpleGamePolicy
    p # dictionary mapping actions to probabilities
    function SimpleGamePolicy(p::Base.Generator)
        return SimpleGamePolicy(Dict(p))
    end

    function SimpleGamePolicy(p::Dict)
        vs = collect(values(p))
        vs ./= sum(vs)
        # return SimpleGamePolicy from dictionary, calculate as action-probability
        return new(Dict(k => v for (k, v) in zip(keys(p), vs)))
    end
    # return SimpleGamePolicy with probability of ai is 1
    SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)  # return probability agent i will do action ai

function (πi::SimpleGamePolicy)()
    D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
    return rand(D)  # return random action
end

joint(X) = vec(collect(Iterators.product(X...)))  # create joint action space from X
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]  # replace π[i] with πi

# function to compute the utility associated with executing joint policy π in the game 𝒫 from the perspective of agent i.
function utility(𝒫::SimpleGame, π, i)
    𝒜, R = 𝒫.𝒜, 𝒫.R

    # probability action a occur
    p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
    # the utility of agent i with joint policy π
    return sum(R(a)[i] * p(a) for a in joint(𝒜))
end