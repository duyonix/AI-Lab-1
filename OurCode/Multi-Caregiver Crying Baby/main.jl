import Pkg;
Pkg.add("Parameters");
using Random
include("BabyPOMG.jl")
include("../helpers/POMG/ConditionalPlan.jl")
include("../helpers/POMG/POMG.jl")
include("../helpers/POMG/POMGDynamicProgramming.jl")
include("../helpers/POMG/POMGNashEquilibrium.jl")
include("../helpers/SimpleGame/SimpleGame.jl")
include("../helpers/SimpleGame/SimpleGamePolicy.jl")
include("../helpers/SimpleGame/NashEquilibrium.jl")
include("FormatAnswer.jl")

function MultiCaregiverCryingBaby()
    BabyPOMDP = CryingBaby()
    return BabyPOMG(BabyPOMDP)
end

function create_conditional_plans(𝒫, d)
    # create all conditional plan with depth d from P::POMG
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
    end
    return Π
end

function expand_conditional_plans(𝒫, Π)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    return [[ConditionalPlan(ai, Dict(oi => πi for oi in 𝒪[i])) for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end



function solve(M::POMGNashEquilibrium, 𝒫::POMG)
    # step 1: convert POMG to SimpleGame
    # step 2: find Nash Equilibrium of that SimpleGame
    ℐ, γ, b, d = 𝒫.ℐ, 𝒫.γ, M.b, M.d
    Π = create_conditional_plans(𝒫, d)
    U = Dict(π => utility(𝒫, b, π) for π in joint(Π))
    𝒢 = SimpleGame(γ, ℐ, Π, π -> U[π])
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

function solve(M::POMGDynamicProgramming, 𝒫::POMG)
    ℐ, 𝒮, 𝒜, R, γ, b, d = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.γ, M.b, M.d
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
        prune_dominated!(Π, 𝒫) # after expand, prune dominated conditional plan
    end
    𝒢 = SimpleGame(γ, ℐ, Π, π -> utility(𝒫, b, π))
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

function prune_dominated!(Π, 𝒫::POMG)
    # prune any policy that is dominated by another policies
    done = false
    while !done
        done = true
        for i in shuffle(𝒫.ℐ)
            for πi in shuffle(Π[i])
                if length(Π[i]) > 1 && is_dominated(𝒫, Π, i, πi)
                    filter!(πi′ -> πi′ ≠ πi, Π[i])
                    done = false
                    break
                end
            end
        end
    end
end

function is_dominated(𝒫::POMG, Π, i, πi)
    # check if policy is dominated
    ℐ, 𝒮 = 𝒫.ℐ, 𝒫.𝒮
    jointΠnoti = joint([Π[j] for j in ℐ if j ≠ i])
    π(πi′, πnoti) = [j == i ? πi′ : πnoti[j > i ? j - 1 : j] for j in ℐ]
    Ui = Dict((πi′, πnoti, s) => evaluate_plan(𝒫, π(πi′, πnoti), s)[i]
              for πi′ in Π[i], πnoti in jointΠnoti, s in 𝒮)
    model = Model(Ipopt.Optimizer)
    @variable(model, δ)
    @variable(model, b[jointΠnoti, 𝒮] ≥ 0)
    @objective(model, Max, δ)
    @constraint(model, [πi′ = Π[i]],
        sum(b[πnoti, s] * (Ui[πi′, πnoti, s] - Ui[πi, πnoti, s])
            for πnoti in jointΠnoti for s in 𝒮) ≥ δ)
    @constraint(model, sum(b) == 1)
    optimize!(model)
    return value(δ) ≥ 0
end


b = [0.5, 0.5] # initial state distribution, b[sated]=b[hungry]=0.5, we can set this to [0.2, 0.8]
d = 3 # depth of conditional plans

# pomgDP=POMGDynamicProgramming(b, 5) # uncomment 2 lines and comment 2 lines to run POMGDP
pomgNash=POMGNashEquilibrium(b, d)
# ans=solve(pomgDP, pomg)
ans = solve(pomgNash, pomg)
printAns(ans)

