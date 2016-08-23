__precompile__(true)

module DatasToolbox


using DataFrames
using PyCall

# explicit imports for overriding base
import Base.serialize
import Base.deserialize
import Base.shuffle!
import Base.convert


# exports are done in files, it's just much easier

include("dfutils.jl")
include("tsutils.jl")
include("utils.jl")
include("DataHandler.jl")
include("TimeSeriesHandler.jl")





end
