using DatasToolbox
using DataFrames
using NullableArrays
using PyCall

df = makeTestDF(Int64, Float64, String, DateTime, Date, nrows=10^6)
df[2, 3] = Nullable()

pydf = pandas(df)

@time newdf = convertPyDF(pydf)

coltypes = [eltype(newdf[c]) for c in names(newdf)]

