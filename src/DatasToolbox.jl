__precompile__(true)

module DatasToolbox


using DataFrames
using PyCall

# explicit imports for overriding base
import Base.serialize
import Base.deserialize


# exports are done in files, it's just much easier

include("dfutils.jl")
include("utils.jl")







end
