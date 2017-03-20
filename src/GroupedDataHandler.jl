#============================================================================
I haven't decided yet whether I actually want to use this
============================================================================#



type GroupedDataHandler{T} <: AbstractDH{T}

    df::DataTable

    colsInput::Vector{Symbol}
    colsOutput::Vector{Symbol}

    colsClass::Vector{Symbol}

    colsNormalize::Vector{Symbol}

    keys::Vector

    mu::Vector{T}
    norm::Vector{T}
    userange::Bool

    dfTrain::DataTable
    dfTest::DataTable

    dfTrain_grp::GroupedDataTable
    dfTest_grp::GroupedDataTable

    X_train::Dict{Any,Array{T}}
    y_train::Dict{Any,Array{T}}
    # TODO we need a way of getting these back into the dataframe
    X_test::Dict{Any,Array{T}}
    y_test::Dict{Any,Array{T}}

    # this is where predictions are kept
    yhat::Array{T}
    yhat_train::Array{T}

    function GroupedDataHandler(df::DataTable, class_cols::Vector{Symbol}; 
                                testfrac::AbstractFloat=0.0, 
                                shuffle::Bool=false,
                                input_cols::Vector{Symbol}=Symbol[],
                                output_cols::Vector{Symbol}=Symbol[],
                                normalize_cols::Vector{Symbol}=Symbol[],
                                assign::Bool=false,
                                userange::Bool=false,
                                compute_keys::Bool=true
                               )
        if sum(!complete_cases(df)) ≠ 0
            throw(ArgumentError("GroupedDataHandler only accepts complete dataframes."))
        end
        ndf = copy(df)
        # TODO convert to all non-nullable arrays!!!
        o = new(ndf, input_cols, output_cols, class_cols, normalize_cols) 
        o.userange = userange
        compute_keys && keys!(o)
        split!(o, testfrac, shuffle=shuffle, assign=assign)
        computeNormalizeParameters!(o, dataset=:dfTrain)
        if canNormalize(o)
            normalizeTrain!(o)
            size(o.dfTest,1) > 0 && normalizeTest!(o)
        end
        o
    end
end
export GroupedDataHandler


function keytuple(df::AbstractDataTable, cols::Vector{Symbol}, idx::Integer=1)
    tuple(convert(Array{Any}, df[idx, cols])...)
end
export keytuple


function keys{T}(gdh::GroupedDataHandler{T})
    if isdefined(gdh, :keys)
        return gdh.keys
    end
    o = mapreduce(vcat, groupby(gdh.df, gdh.colsClass)) do sdf
        [keytuple(sdf, gdh.colsClass)]
    end
    o
end

keys!{T}(gdh::GroupedDataHandler{T}) = (gdh.keys = keys(gdh))
export keys, keys!


function assignTrain!{T}(gdh::GroupedDataHandler{T})
    if isempty(gdh.dfTrain)
        throw(ErrorException("Attempting to assign data from empty training dataframe."))
    end
    gdh.dfTrain_grp = groupby(gdh.dfTrain, gdh.colsClass)
    Xdict, ydict = getMatrixDict(T, gdh.dfTrain_grp, gdh.colsClass, gdh.colsInput, 
                                 gdh.colsOutput)
    gdh.X_train = Xdict
    gdh.y_train = ydict
    Xdict, ydict
end


function assignTest!{T}(gdh::GroupedDataHandler{T})
    if isempty(gdh.dfTest)
        throw(ErrorException("Attempting to assign data from empty test dataframe."))
    end
    gdh.dfTest_grp = groupby(gdh.dfTest, gdh.colsClass)
    Xdict, ydict = getMatrixDict(T, gdh.dfTest_grp, gdh.colsClass, gdh.colsInput, 
                                 gdh.colsOutput)
    gdh.X_test = Xdict
    gdh.y_test = ydict
    Xdict, ydict
end


function getTrainData{T}(gdh::GroupedDataHandler{T}; flatten::Bool=false)
    X, y = gdh.X_train, gdh.y_train
    if flatten
        y = Dict{eltype(keys(y)), Vector{T}}()
        for (k, v) ∈ y
            @assert size(v,2) == 1 "Attempted to flatten rank-2 array."
            y[k] = squeeze(v, 2)
        end
    end
    X, y
end


function getTestData{T}(gdh::GroupedDataHandler{T}; flatten::Bool=false)
    X, y = gdh.X_test, gdh.y_test
    if flatten
        y = Dict{eltype(keys(y)), Vector{T}}()
        for (k, v) ∈ y
            @assert size(v,2) == 1 "Attempted to flatten rank-2 array."
            y[k] = squeeze(v, 2)
        end
    end
    X, y
end


"""
    _fix_flattened_matrix_dict(T, dict)

Takes a dictionary with matrix or vector values and, if they are vectors, converts them
to `Matrix`s with a single columns.
"""
function _fix_flattened_matrix_dict{T,K,V<:Vector}(::Type{T}, dict::Dict{K,V})
    dict_new = Dict{eltype(keys(dict)), Matrix{T}}()
    for (k, v) ∈ dict
        dict_new[k] = reshape(v, (length(v), 1))
    end
    dict = dict_new
end
_fix_flattened_matrix_dict{T,K,V<:Matrix}(::Type{T}, dict::Dict{K,V}) = dict


"""
    _replace_values_into_grouped(gp, dict, T, new_col_names, cols)

Takes a dict of matrices and places them into a grouped dataframe.  The matrices must have
the same number of rows as their respective groups in the grouped dataframe `gp`.  
A new dataframe will be output with rows named according to `new_col_names` with columns
corresponding to the columns of matrices in `dict`, and rows corresponding to the rows
of the original dataframe.  The keys of `dict` should be tuples like those produced by
`keytuple`.  The type parameter `T` denotes the type of the matrices contained in `dict`.

Note that the implmentation on this depends on the "private" members of `GroupedDataTable`.
"""
function _replace_values_into_grouped{T}(gp::GroupedDataTable, dict::Dict, ::Type{T},
                                         new_col_names::Vector{Symbol},
                                         keycols::Vector{Symbol})
    newcols = DataTable([T for n ∈ new_col_names], new_col_names, size(gp.parent, 1))
    for (start, stop) ∈ zip(gp.starts, gp.ends)
        key = keytuple(gp.parent, keycols, gp.idx[start])
        y = get(dict, key, NullableArray(T, length(start:stop), length(new_col_names)))
        for (i, idx) ∈ enumerate(gp.idx[start:stop])
            for j ∈ 1:length(new_col_names)
                newcols[idx, j] = y[i, j]
            end
        end
    end
    newcols
end
                                      

function getTestAnalysisData{T}(gdh::GroupedDataHandler{T}, ŷ::Dict; 
                                names::Vector{Symbol}=Symbol[],
                                squared_error::Bool=true)
    df = copy(gdh.dfTest)

    ŷ = _fix_flattened_matrix_dict(T, ŷ)

    if length(names) == 0
        names = getDefaultTestAnalysisColumnNames(gdh)
    end

    newcols = _replace_values_into_grouped(gdh.dfTest_grp, ŷ, T, names, gdh.colsClass)

    df = hcat(df, newcols)

    # note that it's now possible to have nulls
    for (idx, name) ∈ enumerate(names)
        orig_col = df[gdh.colsOutput[idx]]
        err = df[name] - orig_col
        df[Symbol(string(name, "_Error"))] = err
        if squared_error
            # they haven't written the ^ operator for NullableArrays
            df[Symbol(string(name, "_Error²"))] = err .* err
        end
    end

    df
end


"""
    getGroupedTestAnalysisData(data, keycols[; names=[], squared_error=true])
    getGroupedTestAnalysisData(dh, data, keycols[; names=[], squared_error=true])
    getGroupedTestAnalysisData(gdh, data[; names=[], squared_error=true])
    getGroupedTestAnalysisData(gdh, ŷ[; names=[], squared_error=true])

Groups the output of `getTestAnalysisData` by the columns `keycols`.  This is particularly
useful for `GroupedDataHandler` where a typical use case is applying different estimators
to different subsets of the data.  One can supply the output `getTestAnalysisData` as
`data` or pass a `GroupedDataHandler` together with an output dictionary `ŷ`, in which
case all the tables will be generated for you.
"""
function getGroupedTestAnalysisData(data::DataTable, keycols::Vector{Symbol},
                                    names::Vector{Symbol}; 
                                    squared_error::Bool=true)
    by(data, keycols) do sdf
        agg = DataTable()
        if isempty(sdf) 
            return agg
        end
        agg[:NDataPoints] = size(sdf,1)
        for name ∈ names
            errname = Symbol(string(name, "_Error"))
            agg[Symbol(string(errname, "_mean"))] = mean(dropnull(sdf[errname]))
            agg[Symbol(string(errname, "_std"))] = std(dropnull(sdf[errname]))
            if squared_error
                err2name = Symbol(string(name, "_Error²"))
                agg[Symbol(string(err2name, "_mean"))] = mean(dropnull(sdf[err2name]))
                agg[Symbol(string(err2name, "_std"))] = std(dropnull(sdf[err2name]))
            end
        end
        agg
    end
end

function getGroupedTestAnalysisData(dh::AbstractDH, data::DataTable, keycols::Vector{Symbol};
                                    names::Vector{Symbol}=Symbol[],
                                    squared_error::Bool=true)
    if length(names) == 0
        names = getDefaultTestAnalysisColumnNames(dh)
    end
    getGroupedTestAnalysisData(data, keycols, names, squared_error=squared_error)
end

function getGroupedTestAnalysisData(gdh::GroupedDataHandler, data::DataTable;
                                    names::Vector{Symbol}=Symbol[],
                                    squared_error::Bool=true)
    getGroupedTestAnalysisData(gdh, data, gdh.colsClass, names=names, 
                               squared_error=squared_error)
end

function getGroupedTestAnalysisData(gdh::GroupedDataHandler, ŷ::Dict; 
                                    names::Vector{Symbol}=Symbol[],
                                    squared_error::Bool=true)
    if length(names) == 0
        names = getDefaultTestAnalysisColumnNames(gdh)
    end
    data = getTestAnalysisData(gdh, ŷ, names=names, squared_error=squared_error)
    getGroupedTestAnalysisData(data, gdh.colsClass, names, squared_error=squared_error)
end
export getGroupedTestAnalysisData



