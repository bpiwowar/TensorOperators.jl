using TensorOperators
using Base.Test

@test 1 == 1

l2cost = TensorOperators.L2Cost()

x = rand(100, 100)
z = rand(100, 5)
m = TensorOperators.LinearModule{CPUDouble}(100, 5)
epsilon = 5e-5

@time for i = 1:100
    initGradient(m)

    y = forward(m, x)
    e = forward(l2cost, (y, z))
    println("Cost = $e")

    backward(m, x, backward(l2cost, (y, z)))
    m.bias -= epsilon * m.gradBias
    m.weight -= epsilon * m.gradWeight
end
