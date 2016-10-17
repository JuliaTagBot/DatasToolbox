

"""
    convertPyColumn(pycol::PyObject)

Converts a column of a pandas array to a Julia `NullableArray`.
"""
function convertPyColumn(pycol::PyObject)::NullableArray
    nptype = pycol[:dtype][:kind]
    # list of numpy kinds can be found at 
    # http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
    if nptype == "O"
        o = NullableArray([x == nothing ? Nullable() : x for x in pycol])
        return o
    elseif nptype == "M"
        (o = NullableArray(Nullable{DateTime}[isa(t, PyObject) ? Nullable() : t for t in
                                               pycol]))
        return o
    else
        return NullableArray(pycol[:values])
    end
end
export convertPyColumn


function _inferColumnType(col::NullableArray; max_elements::Integer=100)::DataType
    for i in 1:max_elements
        isnull(col) ? continue : nothing
        thistype = typeof(get(col[i]))
        thistype == PyObject ? continue : nothing
        return thistype
    end
    return Any
end


function _fillNewCol!{T}(newcol::NullableArray{T}, df::DataFrame, col::Symbol)
    for i in 1:length(newcol)
        if isnull(df[i, col])
            newcol[i] = Nullable()
        else
            newcol[i] = get(df[i, col])
        end
    end
end


"""
    fixColumnTypes!(df)

Check to see if the dataframe `df` has any columns of type `Any` and attempt to convert
them to the proper types.  This can be called from `convertPyDF` with the option
`fixtypes`.
"""
function fixColumnTypes!(df::DataFrame)
    for col in names(df)
        if eltype(eltype(df[col])) ≠ Any continue end
        dtype = _inferColumnType(df[col])
        # I can't find any way around getting into these stupid loops
        newcol = NullableArray{dtype}(length(df[col]))
        _fillNewCol!(newcol, df, col)
        df[col] = newcol
    end
end
export fixColumnTypes!


# TODO this still has the fix_nones option for legacy support, remove when
# DataFrames with NullableArrays is in full release
"""
    convertPyDF(pydf[, fixtypes=true])

Converts a pandas dataframe to a Julia one.  

Note that it is difficult to infer the correct types of columns which contain references
to Python objects.  If `fixtypes`, this will attempt to convert any column with eltype
`Any` to the proper type.
"""
function convertPyDF(pydf::PyObject; fixtypes::Bool=true,
                     fix_nones::Bool=false)::DataFrame
    df = DataFrame()
    for col in pydf[:columns]
        df[Symbol(col)] = convertPyColumn(get(pydf, col))
    end
    if fixtypes fixColumnTypes!(df) end
    return df
end
export convertPyDF


"""
    fixPyNones(dtype, a)

Attempts to convert a `NullableArray` to have eltype `dtype` while replacing all Python
`None`s with `Nullable`.
"""
function fixPyNones(dtype::DataType, a::NullableArray)
    # exit silently if the array can't possibly hold Nones
    if !((eltype(a) == Any) | (eltype(a) == PyObject)) return end
    pyNone = pybuiltin("None")
    newa = NullableArray([x == pyNone ? Nullable() : convert(dtype, x) for x in a])
end
export fixPyNones


"""
    fixPyNones!(dtype::DataType, df::DataFrame, col::Symbol)

Attempts to convert a column of the dataframe to have eltype `dtype` while replacing all
Python `None`s with `Nullable()`.
"""
function fixPyNones!(dtype::DataType, df::DataFrame, col::Symbol)
    df[col] = fixPyNones(dtype, df[col])
    return df
end
export fixPyNones!


"""
    fixPyNones!(df::DataFrame)

Attempts to automatically convert all columns of a dataframe to have eltype `Any` while
replacing all Python `None`s with `Nullable()`.
"""
function fixPyNones!(df::DataFrame)
    for col in names(df)
        if eltype(df[col]) == PyObject
            fixPyNones!(Any, df, col)
        end
    end
end
export fixPyNones!


"""
    convert(dtype, a)

This converts a column of floats that should have been ints, but got converted to
floats because it has missing values which were converted to NaN's.
The supplied `NullableArray` should have eltype `Float32` or `Float64`.
"""
function convert(dtype::Union{Type{Int32}, Type{Int64}}, a::NullableArray{Float32})
    NullableArray([isnan(x) ? Nullable() : convert(dtype, x) for x in a])    
end
export convert


function convert(dtype::Union{Type{Int32}, Type{Int64}}, a::NullableArray{Float64, 1})
    newa = NullableArray([isnan(x) ? Nullable() : convert(dtype, x) for x in a])
end


"""
    unpickle([dtype,] filename[, migrate, fix_nones])

Deserializes a python pickle file and returns the object it contains.
Additionally, if `DataFrame` is given as the first argument, will
attempt to convert the object to a Julia dataframe with the flags
`migrate` and `fix_nones` (see `convertPyDF`).
"""
function unpickle(filename::String)::PyObject
    f = pyeval("open('$filename', 'rb')")
    pyobj = PyPickle[:load](f)
end

function unpickle(dtype::Type{DataFrame}, filename::AbstractString;
                  migrate::Bool=true,
                  fix_nones::Bool=true)::DataFrame
    f = pyeval("open('$filename', 'rb')")
    pydf = PyPickle[:load](f)
    df = convertPyDF(pydf, migrate=migrate, fix_nones=fix_nones)
end
export unpickle


"""
    pickle(filename, object)

Converts the provided object to a PyObject and serializes it in
the python pickle format.  If the object provided is a `DataFrame`,
this will first convert it to a pandas dataframe.
"""
function pickle(filename::String, object::Any)
    pyobject = PyObject(object)
    f = pyeval("open('$filename', 'wb')")
    PyPickle[:dump](pyobject, f)
end

function pickle(filename::String, df::DataFrame)
    pydf = pandas(df)
    f = pyeval("open('$filename', 'wb')")
    PyPickle[:dump](pydf, f)
end
export pickle


"""
    shuffle!(df::DataFrame)

Shuffles a dataframe in place.
"""
function shuffle!(df::DataFrame)
    permutation = shuffle(collect(1:size(df)[1]))
    tdf = copyColumns(df)
    for i in 1:length(permutation)
        df[i, :] = tdf[permutation[i], :]
    end
    return df
end
export shuffle!


"""
    numericalCategories(otype::DataType, A::Array)

Converts a categorical variable into numerical values of the given type.

Returns the mapping as well as the new array, but the mapping is just an array
so it always maps to an integer
"""
function numericalCategories(otype::DataType, A::Array)
    mapping = sort!(unique(A))
    o = convert(Array{otype}, indexin(A, mapping))
    return o, mapping
end
# define for NullableArray type
numericalCategories(otype::DataType, A::NullableArray) = numericalCategories(otype, 
        convert(Array, A))
export numericalCategories


"""
    getDefaultCategoricalMapping(A::Array)

Gets the default mapping of categorical variables which would be returned by
numericalCategories.
"""
function getDefaultCategoricalMapping(A::Array)
    return sort!(unique(A))
end
export getDefaultCategoricalMapping


"""
    numericalCategories!(otype::DataType, df::DataFrame, col::Symbol)

Converts a categorical value in a column into a numerical variable of the given
type.

Returns the mapping.
"""
function numericalCategories!(otype::DataType, df::DataFrame, col::Symbol)
    df[Symbol(string(col)*"_Orig")] = df[col]
    df[col], mapping = numericalCategories(otype, df[col])
    return mapping
end
export numericalCategories!


"""
    numericalCategories!(otype::DataType, df::DataFrame, cols::Array{Symbol}) 

Converts categorical variables into numerical values for multiple columns in a
dataframe.  

**TODO** For now doesn't return mapping, may have to implement some type of 
mapping type.
"""
function numericalCategories!(otype::DataType, df::DataFrame, cols::Array{Symbol})
    for col in cols
        numericalCategories!(otype, df, col)
    end
    return
end
export numericalCategories!


"""
    convertNulls!{T}(A::Array{T, 1}, newvalue::T)

Converts all null values (NaN's and Nullable()) to a particular value.
Note this has to check whether the type is Nullable.
"""
function convertNulls!{T <: AbstractFloat}(A::Vector{T}, newvalue::T)
    for i in 1:length(A)
        if isnan(A[i])
            A[i] = newvalue
        end
    end
    return A
end

function convertNulls!{T <: Nullable}(A::Vector{T}, newvalue::T)
    for i in 1:length(A)
        if isnull(A[i])
            A[i] = newvalue
        end
    end
    return A
end

export convertNulls!


"""
    convertNulls{T}(A, newvalue)

Converts all null vlaues (NaN's and Nullable()) to a particular value.
This is a wrapper added for sake of naming consistency.
"""
function convertNulls{T}(A::NullableArray{T}, newvalue::T)
    convert(Array, A, newvalue)
end
export convertNulls


"""
    convertNulls!(df::DataFrame, cols::Vector{Symbol}, newvalue::Any)

Convert all null values in columns of a DataFrame to a particular value.

There is also a method for passing a single column symbol, not as a vector.
"""
function convertNulls!(df::DataFrame, cols::Vector{Symbol}, newvalue::Any)
    for col in cols
        df[col] = convertNulls(df[col], newvalue)
    end
    return
end
convertNulls!(df::DataFrame, col::Symbol, newvalue) = convertNulls!(df, [col], newvalue)
export convertNulls!


"""
    copyColumns(df::DataFrame)

The default copy method for dataframes only copies one level deep, so basically it stores
an array of columns.  If you assign elements of individual (column) arrays then, it can
make changes to references to those arrays that exist elsewhere.

This method instead creates a new dataframe out of copies of the (column) arrays.

This is not named copy due to the fact that there is already an explicit copy(::DataFrame)
implementation in dataframes.

Note that deepcopy is recursive, so this is *NOT* the same thing as deepcopy(df), which 
copies literally everything.
"""
function copyColumns(df::DataFrame)
    ndf = DataFrame()
    for col in names(df)
        ndf[col] = copy(df[col])
    end
    return ndf
end
export copyColumns


# this may be very inefficient due to the way Julia type optimization works
"""
    applyCatConstraints(dict, df[, kwargs])

Returns a copy of the dataframe `df` with categorical constraints applied.  `dict` should 
be a dictionary with keys equal to column names in `df` and values equal to the categorical
values that column is allowed to take on.  For example, to select gauge bosons we can
pass `Dict(:PID=>[i for i in 21:24; -24])`.  Alternatively, the values in the dictionary
can be functions which return boolean values, in which case the returned dataframe will
be the one with column values for which the functions return true.

Note that this requires that the dictionary values are either `Vector` or `Function` 
(though one can of course mix the two types).

Alternatively, instead of passing a `Dict` one can pass keywords, for example
`applyCatConstraints(df, PID=[i for i in 21:24; -24])`.
"""
function applyCatConstraints(dict::Dict, df::DataFrame)
    constr = Bool[true for i in 1:size(df)[1]]
    for (col, values) in dict
        constr &= if typeof(values) <: Vector
            convert(BitArray, map(x -> x ∈ values, df[col]))
        elseif typeof(values) <: Function
            convert(BitArray, map(values, df[col]))
        else
            throw(ArgumentError("Constraints must be either vectors or functions."))
        end
    end
    return df[constr, :]
end

function applyCatConstraints(df::DataFrame; kwargs...)
    dct = Dict(kwargs)
    applyCatConstraints(dct, df)
end
export applyCatConstraints


"""
    pandas(df)

Convert a dataframe to a pandas pyobject.
"""
function pandas(df::DataFrame)::PyObject
    pydf = PyPandas[:DataFrame]()
    for col in names(df)
        pycol = [isnull(x) ? nothing : get(x) for x in df[col]]
        set!(pydf, string(col), pycol)
        # convert datetime to proper numpy type
        if eltype(df[col]) == DateTime
            set!(pydf, string(col), 
                 get(pydf, string(col))[:astype]("<M8[ns]"))
        end
    end
    return pydf
end
export pandas


# TODO there were some bizarre errors when doing this with map, may be inefficient
"""
    constrainDF(df, constraints)
    constrainDF(df, kwargs...)

Returns a constrained dataframe.  For each key `k` in `constraints`, only the elements
of `df` with `df[i, k] ∈ constraints[k]` will be present in the constrained dataframe.

Alternatively, one can provide the columns and lists of acceptable arguments as kewords
of `constrainDF`.

Ideally, the values should be provided as Arrays or other iterables, but in many cases
single values will work.

Note that this will be replaced by DataFramesMeta once it is more mature.
"""
function constrainDF(df::DataFrame, constraints::Dict)::DataFrame
    inrow = ones(Bool, size(df, 1))
    for (col, values) in constraints
        for i in 1:length(inrow)
            if !isnull(df[i, col])
                inrow[i] &= get(df[i, col]) ∈ values 
            end
        end
    end
    df[inrow, :]
end


function constrainDF(df::DataFrame; kwargs...)
    dict = Dict(kwargs)
    constrainDF(df, dict)
end
export constrainDF


# TODO this should only be temporary!!!
"""
    writePandasFeather(filename, df)

Writes the dataframe `df` to a feather by first converting to python.
Note that this is only a temporary solution while Feather.jl matures.

Note that `feather` must be installed in Python3.
"""
function writePandasFeather(filename::String, df::DataFrame)
    pydf = pandas(df)
    PyFeather[:write_dataframe](pydf, filename)
end
export writePandasFeather


"""
    makeTestDF(dtypes...; nrows=10^4)

Creates a random dataframe with columns of types specified by `dtypes`.  This is useful
for testing various dataframe related functionality.
"""
function makeTestDF(dtypes::DataType...; nrows::Integer=10^4)::DataFrame
    df = DataFrame()
    for (idx, dtype) in enumerate(dtypes)
        col = Symbol(string(dtype)*string(idx))
        if dtype <: Real
            df[col] = rand(dtype, nrows)
        elseif dtype <: AbstractString
            df[col] = [randstring(rand(8:16)) for i in 1:nrows]
        elseif dtype <: Dates.TimeType
            df[col] = [dtype(now()) + Dates.Day(i) for i in 1:nrows]
        elseif dtype <: Symbol
            df[col] = [Symbol(randstring(rand(4:12))) for i in 1:nrows]
        end
    end
    return df
end
export makeTestDF


"""
    featherWrite(filename, df[, overwrite=false])

A wrapper for writing dataframes to feather files.  To be used while Feather.jl package
is in development.

If `overwrite`, this will delete the existing file first (an extra step taken to avoid some
strange bugs).
"""
function featherWrite(filename::AbstractString, df::DataFrame;
                      overwrite::Bool=false)::Void
    if isfile(filename)
        if !overwrite
            throw(SystemError("File already exists.  Use overwrite=true."))
        end
        rm(filename)     
    end
    Feather.write(filename, df)
    return nothing
end
export featherWrite


"""
    featherRead(filename)

A wrapper for reading dataframes which are saved in feather files.  To be used while the
Feather.jl package is in development.
"""
function featherRead(filename::AbstractString)::DataFrame
    bad_df = Feather.read(filename)
    df = DataFrame()
    for col in names(bad_df)
        if eltype(bad_df[col]) == Nullable{Feather.WeakRefString{UInt8}}
            df[col] = convert(Array{String}, bad_df[col])
        else
            df[col] = bad_df[col]
        end
    end
    return df
end
export featherRead


