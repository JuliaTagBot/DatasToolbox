using DataFrames
using DatasToolbox

const SEQ_LENGTH = 3

df = DataFrame(A=1:10, B=101:110, C=201:210)

dh = TimeSeriesHandler{Int64}(df, :A, SEQ_LENGTH, input_cols=[:B, :C], output_cols=[:B])

split!(dh, 3)
assign!(dh)

function predict(X)
    println(size(X))
    println(X)
    [X[1, end, 1] + 201]
end

function f(X)
    println("$(size(X)) being called!!!")
    Dict(2=>(X[2] + 1001))
end


info("generating sequence")
# seq = generateSequence(predict, dh, 3, on_matrix=false)
seq = generateSequence(predict, dh, 3, f, on_matrix=false)
info("done generating sequence")


