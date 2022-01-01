import Pkg;
Pkg.add("Parameters");
using Random
include("BabyPOMG.jl")
include("../helpers/POMG/ConditionalPlan.jl")
include("../helpers/POMG/POMG.jl")
include("../helpers/POMG/POMGDynamicProgramming.jl")
include("../helpers/POMG/POMGNashEquilibrium.jl")
include("../helpers/SimpleGame/SimpleGame.jl")
include("../helpers/SimpleGame/NashEquilibrium.jl")
include("FormatAnswer.jl")
include("EvaluateAnswer.jl")

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

b = [0.2, 0.8] # initial state distribution, b[sated]=0.8, b[hungry]=0.2, we can set this to [0.5, 0.5]
d = 3 # depth of conditional plans
multicaregiver_cryingbaby = MultiCaregiverCryingBaby() # return instance babyPOMG
pomg = POMG(multicaregiver_cryingbaby) # return POMG instance from babyPOMG instance
# pomgDP=POMGDynamicProgramming(b, 5) # uncomment 2 lines and comment 2 lines to run POMGDP
pomgNash = POMGNashEquilibrium(b, d)
# ans=solve(pomgDP, pomg)
ans = solve(pomgNash, pomg)
printAns(ans)
C1 = []
C2 = []
createVector!(ans[1], C1, 1)
createVector!(ans[2], C2, 1)
# evaluate_answer(C1, C3, C2, 1)

function test(C1, C2)
    # caregiver 1 là C1, caregiver 2 là C2
    # em sẽ generate 1000 conditional plan đối đầu với C1 để xem C1 thắng được bao nhiêu lần
    c1Wins = 0
    n = length(C1)
    C3 = []
    for i = 1:1000
        empty!(C3)
        for j = 1:n
            push!(C3, [])
            m = length(C1[j])
            for k = 1:m
                temp = rand(Int)
                if temp % 2 == 0
                    push!(C3[j], FEED)
                else
                    push!(C3[j], IGNORE)
                end
            end
        end
        win = evaluate_answer(C1, C3, C2, 1)
        if win
            c1Wins = c1Wins + 1
            # println("WIN")
        end
    end
    print("C1 wins: ")
    println(c1Wins)
end

test(C1, C2)

