using Luxor, Colors
#Some function
# sethue("colorname")         => set color
# background("colorname")     => set background color
# ngon(position, radius, edges, angle, :attribute)  => draw function
# text(string, position,angle, :attribute)          => text on drawing

function drawHexWorld(numCells, emptyCell, actionsEachCell, neReward, poReward)
    directionArrow = "→"
    Drawing(600, 200)
    radius = 30
    grid = GridHex(Point(50, 50), radius, 550)
    sethue("red")
    background("white")
    for i in 1:numCells
        sethue("darkturquoise")
        if i in emptyCell
            sethue("white")
        end
        p = nextgridpoint(grid)
        if i in neReward
            setopacity(0.5)
            sethue("red")
            ngon(p, radius - 3, 6, pi / 2, :fillstroke)
        elseif i in poReward
            setopacity(0.7)
            sethue("blue")
            ngon(p, radius - 3, 6, pi / 2, :fillstroke)
        else
            setopacity(1)
            ngon(p, radius - 3, 6, pi / 2, :stroke)
        end
        sethue("black")
        fontsize(25)
        if actionsEachCell[i] != 0
            text(string(directionArrow), p + pointDirection[actionsEachCell[i]], angle = actionsDirection[actionsEachCell[i]] * pi, halign = :center)
        end
    end

    finish()
    preview()
end

function drawHexMap(numCells, emptyCell, actionsEachCell, neReward, poReward)
    directionArrow = "→"
    Drawing(600, 200)
    radius = 30
    grid = GridHex(Point(50, 50), radius, 550)
    sethue("red")
    background("white")
    pre = -2
    rePre = false
    countPre = 0
    for i in 1:numCells
        sethue("darkturquoise")
        if i in emptyCell
            sethue("white")
        end
        p = nextgridpoint(grid)
        if i in neReward
            setopacity(0.5)
            sethue("red")
            ngon(p, radius - 3, 6, pi / 2, :fillstroke)
        elseif i in poReward
            setopacity(0.7)
            sethue("blue")
            ngon(p, radius - 3, 6, pi / 2, :fillstroke)
        else
            setopacity(1)
            ngon(p, radius - 3, 6, pi / 2, :stroke)
        end
        sethue("black")
        fontsize(18)

        if mod1(i, 10) == 10
            suf = 3 - floor(Int, i / 10)
        else
            suf = 2 - floor(Int, i / 10)
        end
        if suf - floor(Int, i / 10) == 2
            rePre = true
        end
        if countPre == 10
            pre = 0
            countPre = 0
        else
            pre = pre + 1
        end

        countPre += 1
        if actionsEachCell[i] != 0
            text(string(pre, ',', suf), p + (0, 7), halign = :center)
        end
    end
    finish()
    preview()
end

function drawReward(numCells, emptyCell, actionsEachCell, neReward, poReward, rewards)
    directionArrow = "→"
    Drawing(600, 200)
    radius = 30
    grid = GridHex(Point(50, 50), radius, 550)
    sethue("red")
    background("white")
    for i in 1:numCells
        sethue("darkturquoise")
        if i in emptyCell
            sethue("white")
        end
        p = nextgridpoint(grid)
        if i in neReward
            setopacity(0.5)
            sethue("red")
            ngon(p, radius - 3, 6, pi / 2, :fillstroke)
        elseif i in poReward
            setopacity(0.7)
            sethue("blue")
            ngon(p, radius - 3, 6, pi / 2, :fillstroke)
        else
            setopacity(1)
            ngon(p, radius - 3, 6, pi / 2, :stroke)
        end
        sethue("black")
        fontsize(18)
        re = Int(get(rewards, i, 0))
        if actionsEachCell[i] != 0
            if re != 0
                text(string(re), p + (0, 7), halign = :center)
            end
        end
    end
    finish()
    preview()
end

function drawAction()
    Drawing(215, 215)
    radius = 30
    grid = GridHex(Point(100, 100), radius, 550)
    sethue("red")
    background("white")
    sethue("darkturquoise")
    p = nextgridpoint(grid)
    ngon(p, radius - 3, 6, pi / 2, :stroke)
    finish()
    preview()
end

function drawActionDirection(numCells, emptyCell, actionsEachCell, neReward, poReward)
    directionArrow = "→"
    Drawing(400, 100)
    radius = 35
    grid = GridHex(Point(50, 50), radius, 550)
    sethue("red")
    background("white")
    for i in 1:6
        sethue("darkturquoise")
        if i in emptyCell
            sethue("white")
        end
        p = nextgridpoint(grid)
        ngon(p, radius - 5, 6, pi / 2, :stroke)
        sethue("black")
        fontsize(25)
        text(string(directionArrow), p + pointDirection[i], angle = actionsDirection[i] * pi, halign = :center)
    end
    finish()
    preview()
end

numCells = 30
emptyCell = [14, 16, 17, 20, 25, 29]
#Position format
pointDirection = [(8, 8), (8, -6), (3, -12), (-8, -4), (-8, 3), (-4, 10)]
#Arrow angle 
actionsDirection = [0, -1 / 3, -2 / 3, 1, 2 / 3, 1 / 3]

rewards = Dict{Float64,Float64}(
    11 => 5.0,
    23 => -10.0,
    30 => 10.0,)


#This result from iteration in solve function
actionsEachCell1 = [6, 5, 4, 1, 4, 4, 4, 6, 5, 5, 1, 4, 3, 0, 6, 0, 0, 1, 6, 0, 2, 3, 1, 3, 0, 1, 4, 2, 0, 1]
actionsEachCell2 = [6, 5, 5, 4, 1, 5, 1, 6, 6, 5, 1, 4, 3, 0, 2, 0, 0, 1, 6, 0, 2, 3, 1, 3, 0, 1, 1, 2, 0, 1]
actionsEachCell3 = [6, 5, 5, 4, 4, 1, 1, 1, 6, 5, 1, 4, 3, 0, 2, 0, 0, 1, 6, 0, 2, 3, 1, 3, 0, 1, 1, 2, 0, 1]
actionsEachCell4 = [6, 5, 5, 4, 1, 1, 1, 1, 6, 5, 1, 4, 3, 0, 2, 0, 0, 1, 6, 0, 2, 3, 1, 3, 0, 1, 1, 2, 0, 1]
actionsEachCell = [6, 5, 5, 4, 1, 1, 1, 1, 6, 5, 1, 4, 3, 0, 2, 0, 0, 1, 6, 0, 2, 3, 1, 3, 0, 1, 1, 2, 0, 1] #final iteration ~ result

drawHexMap(numCells, emptyCell, actionsEachCell, neReward, poReward)
drawAction()
drawReward(numCells, emptyCell, actionsEachCell, neReward, poReward, rewards)
drawActionDirection(numCells, emptyCell, actionsEachCell, neReward, poReward)

function visualizeResult()
    drawHexWorld(numCells, emptyCell, actionsEachCell4, neReward, poReward)
end
visualizeResult()
