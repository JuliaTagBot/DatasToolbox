using DatasToolbox
using DataFrames

df = DataFrame(100*rand(100, 4))

dh = DataHandler{Float64}(df, input_cols=[:x1, :x2], output_cols=[:x3],
                          normalize_cols=[:x1, :x2],
                          testfrac=0.1, shuffle=true, userange=true)



