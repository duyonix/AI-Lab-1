module DecisionMakingProblems

using Distributions
using Parameters
using Random
using LinearAlgebra
using GridInterpolations
using Parameters
using Statistics
using Printf

export
    MDP, HexWorld, StraightLineHexWorld,
    SimpleGame, PrisonersDilemma, RockPaperScissors, Travelers


import Base: <, ==, rand, vec

include("Hex world/hexworld.jl")
include("Multi-Caregiver Crying Baby/main.jl")
include("Predator-Prey Hex World/main.jl")

include("Rock-Paper-Scissors/rockpaperscissors.jl")
include("Traveler's Dilemma/main.jl")



end # module
