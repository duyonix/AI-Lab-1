using Pkg  # Package to install new packages

# Install packages 

using DataFrames
using GLM
using CSV

const global ROCK = 1
const global PAPER = 2
const global SCISSORS = 3


df = CSV.read("trainingFirst.csv", DataFrame)
show(df, allcols = true)


convert_value = Dict(1 => "Rock", 2 => "Paper", 3 => "Scissors")
victory_dict = Dict(ROCK => PAPER, PAPER => SCISSORS, SCISSORS => ROCK)
print("\n")
println("ROCK-PAPER-SCISSORS GAME WITH AI PREDICTION")

last_response = ROCK
second_to_last_response = ROCK
third_to_last_response = ROCK
correct_ai_response = PAPER

ai_victory = 0
user_victory = 0
rounds = 1

linearReg = lm(@formula(correctAiResponse ~ lastResponse + secondToLastResponse + thirdToLastResponse), df)

function ProcessPrediction(data)
    data = trunc(Int, data) # convert to int

    if (data >= ROCK && data <= SCISSORS)
        return data
    end
    # maybe data over 3, return 1 is correct
    return ROCK
end


function VictoryCounter(user, ai)
    global victory_dict
    global ai_victory
    global user_victory
    if (user == ai)
        return 0
    elseif (victory_dict[user] != ai)
        user_victory += 1
        return -1
    else
        ai_victory += 1
        return 1
    end
end

function TrainModel()
    global df
    global linearReg
    print(linearReg)
    linearReg = lm(@formula(correctAiResponse ~ lastResponse + secondToLastResponse + thirdToLastResponse), df)

end

function AiPredict(data_1, data_2, data_3)
    global df
    global linearReg

    global predicts = predict(linearReg, DataFrame(lastResponse = Int64[data_1], secondToLastResponse = Int64[data_2], thirdToLastResponse = Int64[data_3]))
    print(predicts)
    return predicts[1]

end
function run()

    global third_to_last_response
    global second_to_last_response
    global last_response
    global correct_ai_response
    global user_victory
    global ai_victory
    global rounds

    print("(1: Rock, 2: Paper, 3: Scissors) => ")
    print("You choose: ")
    user_response = parse(Int, readline())
    while user_response < 1 || user_response > 3
        print("Invalid number. Please choose again in (1,2,3): ")
        user_response = parse(Int, readline())
    end

    third_to_last_response = second_to_last_response
    second_to_last_response = last_response
    last_response = user_response

    ai_response_raw = AiPredict(last_response, second_to_last_response, third_to_last_response)
    ai_response = ProcessPrediction(ai_response_raw)

    # Increment score based on who won
    result = VictoryCounter(user_response, ai_response)

    println("\n\tAI says: $(convert_value[ai_response])")
    println("\tYou say: $(convert_value[user_response])")

    if (result == 0)
        println("\t--> DRAW ")
    elseif (result == -1)
        println("\t--> YOU WIN ")
    else
        println("\t--> AI WIN ")
    end

    println("\nRound: $rounds")
    println("Score: AI - $ai_victory , You - $user_victory")

    println("------------------------\n")

    correct_ai_response = victory_dict[user_response]

    push!(df, [last_response, second_to_last_response, third_to_last_response, correct_ai_response])

    TrainModel()
    rounds += 1





end

while rounds < 25
    run()
end