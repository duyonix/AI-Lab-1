import Pkg
include("../DecisionMakingProblems.jl")
Pkg.add("JuMP")
using Pkg # Package to install new packages
Pkg.add("Test")
Pkg.add("LinearAlgebra")

using Test
using DecisionMakingProblems
using LinearAlgebra

const p = DecisionMakingProblems

@testset "rock_paper_scissors.jl" begin
    m = RockPaperScissors()
    @test p.n_agents(m) == 2
    @test length(p.ordered_actions(m, rand(1:2))) == 3 && length(p.ordered_joint_actions(m)) == 9
    @test p.n_actions(m, rand(1:2)) == 3 && p.n_joint_actions(m) == 9
    @test -1.0 <= p.reward(m, rand(1:2), [rand(p.ordered_actions(m, 0)), rand(p.ordered_actions(m, 0))]) <= 1.0
    @test [-1.0, -1.0] <= p.joint_reward(m, [rand(p.ordered_actions(m, 0)), rand(p.ordered_actions(m, 0))]) <= [1.0, 1.0]
    simplegame = p.SimpleGame(m)
end
