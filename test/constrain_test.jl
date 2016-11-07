using DataFrames
using DatasToolbox

const NROWS = 5*10^6

df = DataFrame(A=rand(NROWS), B=rand(1:100, NROWS))

const M = 0.5

# testfunc(a, b) = a < M && b % 3 == 0
# testfunc(a) = a < M

# @time ocdf = DatasToolbox.constrain_OLD(df, [:A, :B], testfunc)
# @time cdf = constrain(df, [:A, :B], testfunc)

dict = Dict()
expr = :((:A .< M) && (:B % 3 .== 0))
DatasToolbox._checkConstraintExpr!(expr, dict)


mac = macroexpand(:(@constrain(df, (:A .< M) && (:B % 3 .== 0))))

@time cdf = @constrain(df, (:A .< M) && (:B % 3 .== 0))

@time ccdf = constrain(df, A=(a -> a < M), B=(b -> b % 3 == 0))

