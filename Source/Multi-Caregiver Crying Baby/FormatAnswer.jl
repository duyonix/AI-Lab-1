include("../helpers/POMG/ConditionalPlan.jl")

vectorAns = []

function createVector!(c::ConditionalPlan, vectorAns, i)
    if (length(vectorAns) < i)
        push!(vectorAns, [])
    end
    push!(vectorAns[i], c.a)
    for (key, value) in c.subplans
        createVector!(value, vectorAns, i + 1)
    end
end

function printSpace(n)
    for i = 1:n
        print(" ")
    end
end

function printVectorAns(vectorAns)
    n = length(vectorAns)
    powerOf2 = [2^(n - i + 1) for i in 1:n]
    println("")
    for i = 1:length(vectorAns)
        printSpace(powerOf2[i] / 2 - 1)
        for j = 1:length(vectorAns[i])
            item = vectorAns[i][j]
            if item == FEED
                print("F")
            elseif item == SING
                print("S")
            else
                print("I")
            end
            if j != length(vectorAns[i])
                printSpace(powerOf2[i] - 1)
            end
        end
        println("")
    end
end

function printAns(ans)
    i = 1
    for res in ans
        print("Care-giver ")
        print(i)
        println("'s policy:")
        createVector!(res, vectorAns, 1)
        printVectorAns(vectorAns)
        empty!(vectorAns)
        i = i + 1
    end
end

function printSomething(a)
    for b in a
        for c in b
            println("")
            empty!(vectorAns)
            createVector!(c, vectorAns, 1)
            printVectorAns(vectorAns)
            println("")
        end
    end
end