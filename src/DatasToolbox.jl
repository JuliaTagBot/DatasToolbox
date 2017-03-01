__precompile__(true)

module DatasToolbox


using Reexport

@reexport using DataFrames

using DataFramesMeta
using Feather
using PyCall


# explicit imports for overriding base
import Base.serialize
import Base.deserialize
import Base.shuffle!
import Base.convert
import Base.normalize!
import Base.Dict
import Base.keys

# the following are python imports that get used in various places
const PyPickle = PyNULL()
const PyPandas = PyNULL()
const PyFeather = PyNULL()

function __init__()
    copy!(PyPickle, pyimport("pickle"))
    copy!(PyPandas, pyimport("pandas"))
    copy!(PyFeather, pyimport("feather"))
end


# exports are done in files, it's just much easier

include("dfutils.jl")
include("tsutils.jl")
include("utils.jl")
include("DataHandler.jl")
include("TimeSeriesHandler.jl")
include("GroupedDataHandler.jl")
include("AutoSer.jl")





end
