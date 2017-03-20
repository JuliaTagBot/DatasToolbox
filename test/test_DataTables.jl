using DatasToolbox
import DataFrames



dt = randomData(Int64, Float64, String, nrows=10) 
dt[1, 1] = Nullable()
dt[2, 2] = Nullable()
dt[3, 3] = Nullable()

df = convert(DataFrames.DataFrame, dt)

dtPrime = convert(DataTable, df)

