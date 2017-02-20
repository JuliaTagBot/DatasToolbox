
"""
    AbstractDH{T}

Abstract base class for data handler objects.
"""
abstract AbstractDH{T} <: Any
export AbstractDH


"""
    DataHandler{T} <: AbstractDH{T}

Type for handling datasets.  This is basically a wrapper for a dataframe with methods
for splitting it into training and test sets and creating input and output numerical
arrays.  It is intended that most reformatting of the dataframe is done before passing
it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.
"""
type DataHandler{T} <: AbstractDH{T}

    # dataframe where all data is kept
    df::DataFrame

    colsInput::Vector{Symbol}
    colsOutput::Vector{Symbol}

    colsNormalize::Vector{Symbol}

    mu::Vector{T}
    norm::Vector{T}
    userange::Bool

    # training data, is a subset of df
    dfTrain::DataFrame
    # testing data, is a subset of df
    dfTest::DataFrame

    # these are the arrays of training, testing data
    X_train::Array{T}
    y_train::Array{T}
    X_test::Array{T}
    y_test::Array{T}

    # this is where predictions are kept
    yhat::Array{T}
    yhat_train::Array{T}

    """
        DataHandler{T}(df::DataFrame; testfrac::AbstractFloat=0.0, shuffle::Bool=false,
                    input_cols::Array{Symbol}=Symbol[], output_cols::Array{Symbol}=Symbol[],
                    normalize_cols::Array{Symbol}=Symbol[], assign::Bool=false,
                    userange::Bool=false)

    Constructor for a `DataHandler` object.  The resulting object will produce machine
    learning datasets from the provided dataframe.  The columns for machine learning input
    and output (target) should be provided with keywords `input_cols` and `output_cols`
    respectively.  The constructor will randomly do a train-test split with test fraction
    `testfrac`.  

    Note that currently `DataHandler` can't handle dataframes with null values, and
    automatically drops all rows containing null values.
    """
    function DataHandler(df::DataFrame; testfrac::AbstractFloat=0.0, shuffle::Bool=false,
                         input_cols::Vector{Symbol}=Symbol[], 
                         output_cols::Vector{Symbol}=Symbol[],
                         normalize_cols::Vector{Symbol}=Symbol[],
                         assign::Bool=false,
                         userange::Bool=false)
        if sum(!complete_cases(df)) ≠ 0
            throw(ArgumentError("DataHandler only accepts complete dataframes."))
        end
        ndf = copy(df)
        # TODO convert to all non-nullable arrays!!!
        o = new(ndf, input_cols, output_cols, normalize_cols)
        o.userange = userange
        split!(o, testfrac, shuffle=shuffle, assign=assign)
        computeNormalizeParameters!(o, dataset=:dfTrain)
        if canNormalize(o)
            normalizeTrain!(o)
            size(o.dfTest,1) > 0 && normalizeTest!(o)
        end
        o
    end
end
export DataHandler


# TODO all of this normalization stuff should be on the estimator side. 
# get rid of it and make a good framework for doing this with estimators
"""
    computeNormalizeParameters!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)

Gets the parameters for centering and rescaling from either the training dataset 
(`dataset=:dfTrain`) or the test dataset (`dataset=:dfTest`).

Does this using the training dataframe by default, but can be set to use test.
Exits normally if this doesn't need to be done for any columns.

This should always be called before `normalize!`, that way you have control over what
dataset the parameters are computed from.
"""
function computeNormalizeParameters!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
    if length(dh.colsNormalize)==0 return end
    df = getfield(dh, dataset)
    mu = Array{T, 1}(length(dh.colsNormalize))
    norm = Array{T, 1}(length(dh.colsNormalize))
    for (i, col) in enumerate(dh.colsNormalize)
        dfcol = dropnull(df[col])
        mu[i] = mean(dfcol)
        if dh.userange
            norm[i] = maximum(dfcol) - minimum(dfcol)
        else
            norm[i] = std(dfcol)
        end
    end
    dh.mu = mu
    dh.norm = norm
    return
end
export computeNormalizeParameters!


"""
    canNormalize(dh::AbstractDH)

Determines whether the data in the datahandler can be normlized, i.e. because
the parameters have or haven't been computed yet.
"""
function canNormalize(dh::AbstractDH)
    can = isdefined(dh, :mu) && isdefined(dh, :norm)
    can = can && length(dh.mu) == length(dh.colsNormalize)
    can = can && length(dh.norm) == length(dh.colsNormalize)
    return can
end


"""
    normalize!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
    normalizeTrain!(dh::AbstractDH)
    normalizeTest!(dh::AbstractDH)

Centers and rescales the columns set by `normalize_cols` in the `DataHandler` constructor.
"""
function normalize!{T}(dh::AbstractDH{T}; dataset::Symbol=:dfTrain)
    if !canNormalize(dh)
        error("Trying to normalize before parameters have been computed.")
    end
    df = getfield(dh, dataset)
    # note that this does something because these are not deep copies
    for (i, col) in enumerate(dh.colsNormalize)
        dfcol = dropnull(df[col])
        dfcol = (dfcol - dh.mu[i])./dh.norm[i]
    end
    return
end
export normalize!
normalizeTrain!(dh::AbstractDH) = normalize!(dh, dataset=:dfTrain)
export normalizeTrain!
normalizeTest!(dh::AbstractDH) = normalize!(dh, dataset=:dfTest)
export normalizeTest!


"""
    unnormalize!{T}(dh::AbstractDH{T}, X::Matrix{T}, cols::Vector{Symbol})

Performs the inverse of the centering and rescaling operations on a matrix.
This can also be called on a single column with a `Symbol` as the last argument.
"""
function unnormalize!{T}(dh::AbstractDH{T}, X::Matrix{T}, cols::Vector{Symbol})
    err = "Array must have same number of dimensions as are to be inverted."
    @assert size(X)[2] == length(cols) err
    for (i, col) in enumerate(cols)
        idx = find(dh.colsNormalize .== col)[1]
        X[:, i] = dh.norm[idx]*X[:, i] + dh.mu[idx]
    end
end
export unnormalize!

function unnormalize!{T}(dh::AbstractDH{T}, X::Array{T, 2}, col::Symbol)
    unnormalize!(dh, X, [col])
end
export unnormalize!


"""
    shuffle!(dh::AbstractDH)

Shuffles the main dataframe of the DataHandler.
"""
shuffle!(dh::AbstractDH) = shuffle!(dh.df)
export shuffle!


"""
    assignTrain!(dh::AbstractDH)

Assigns the training data in the data handler so it can be retrieved in proper form.
"""
function assignTrain!{T}(dh::AbstractDH{T})
    if isempty(dh.dfTrain)
        throw(ErrorException("Attempting to assign data from empty training dataframe."))
    end
    X = convert(Array{T}, dh.dfTrain[:, dh.colsInput])
    y = convert(Array{T}, dh.dfTrain[:, dh.colsOutput])
    dh.X_train = X
    dh.y_train = y
    X, y
end
export assignTrain!


"""
    assignTest!(dh::AbstractDH)

Assigns the test data in the data handler.
"""
function assignTest!{T}(dh::AbstractDH{T})
    if isempty(dh.dfTest)
        throw(ErrorException("Attempting to assign data from empty test dataframe."))
    end
    X = convert(Array{T}, dh.dfTest[:, dh.colsInput])
    y = convert(Array{T}, dh.dfTest[:, dh.colsOutput])
    dh.X_test = X
    dh.y_test = y
    X, y
end
export assignTest!


"""
    assign!(dh::AbstractDH)

Assigns training and test data in the data handler.
"""
function assign!{T}(dh::AbstractDH{T})
    assignTrain!(dh)
    assignTest!(dh)
    return
end
export assign!


"""
    getTrainData(dh::AbstractDH; flatten::Bool=false)

Gets the training data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.
"""
function getTrainData(dh::AbstractDH; flatten::Bool=false)
    X, y = dh.X_train, dh.y_train
    if flatten
        @assert size(y,2) == 1 "Attempted to flatten rank-2 array."
        y = squeeze(y, 2)
    end
    return X, y    
end
export getTrainData


"""
    getTestData(dh::AbstractDH; flatten::Bool=false)

Gets the test data input, output tuple `X, y`.

If `flatten`, attempts to flatten `y`.
"""
function getTestData(dh::AbstractDH; flatten::Bool=false)
    X, y = dh.X_test, dh.y_test
    if flatten
        @assert size(y)[2] == 1 "Attempted to flatten rank-2 array."
        y = squeeze(y, 2)
    end
    return X, y
end
export getTestData


"""
    split!(dh::AbstractDH, index::Integer; assign::Bool=true)

Creates a train, test split by index.  The index given is the last index of the training set.
If `assign`, this will assign the training and test data.
"""
function split!(dh::AbstractDH, index::Integer; assign::Bool=true)
    @assert index ≤ size(dh.df)[1] "Index value too large for dataframe."
    dh.dfTrain = dh.df[1:index, :]
    if index == size(dh.df)[1]
        dh.dfTest = DataFrame()
    else
        dh.dfTest = dh.df[(index+1):end, :]
    end
    assign && assign!(dh)
    nothing
end
export split!


"""
    split!(dh::AbstractDH, testfrac::AbstractFloat; shuffle::Bool=false,
           assign::Bool=true)

Creates a train, test split by fraction.  The fraction given is the test fraction.
"""
function split!(dh::AbstractDH, testfrac::AbstractFloat; shuffle::Bool=false,
                assign::Bool=true)
    @assert testfrac ≤ 1.0 "Test fraction must be ∈ [0, 1]."
    if shuffle shuffle!(dh) end
    index = convert(Int64, round((1. - testfrac)*size(dh.df)[1]))
    split!(dh, index, assign=assign)
    nothing
end
export split!


"""
    split!(dh::AbstractDH, constraint::BitArray)

Splits the data into training and test sets using a BitArray that must correspond to elements
of dh.df.  The elements of the dataframe for which the BitArray holds 1 will be in the test 
set, the remaining elements will be in the training set.
"""
function split!(dh::AbstractDH, constraint::BitArray)
    dh.dfTest = dh.df[constraint, :]
    dh.dfTrain = dh.df[!constraint, :]
end
export split!


"""
    @selectTrain!(dh, expr)

Set the training set to be the subset of the `DataHandler`'s dataframe for which `expr` is
true.  `expr` should be an expression which evaluates to a `Bool` with `Symbol`s corresponding
to column names for values.  See the documentation for `@constrain` and `@split!` 
for examples.
"""
macro selectTrain!(dh, expr)
    esc(:($dh.dfTrain = @constrain($dh.df, $expr)))
end
export @selectTrain!


"""
    @selectTrain!(dh, expr)

Set the test set to be the subset of the `DataHandler`'s dataframe for which `expr` is
true.  `expr` should be an expression which evaluates to a `Bool` with `Symbol`s corresponding
to column names for values.  See the documentation for `@constrain` and `@split!` 
for examples.
"""
macro selectTest!(dh, expr)
    esc(:($dh.dfTest = @constrain($dh.df, $expr)))
end
export @selectTest!


"""
    @split!(dh, expr)

Splits the `DataHandler`'s DataFrame into training in test set, such that the test set is the
set of datapoints for which `expr` is true, and the training set is the set of datapoints for
which `expr` is false.  `expr` should be an expression that evaluates to `Bool` with symbols
in place of column names.  For example, if the columns of the dataframe are `[:x1, :x2, :x3]`
one can do `expr = (:x1 > 0.0) | (tanh(:x3) > e^6)`.

See documentation on the `@constrain` macro which `@split!` calls internally.
"""
macro split!(dh, expr)
    esc(quote
        $dh.dfTest  = @constrain($dh.df, $expr)
        $dh.dfTrain = @constrain($dh.df, !($expr))
    end)
end
export @split!


"""
## `DataHandler`

    getTestAnalysisData(dh::AbstractDH, ŷ::Array; names::Vector{Symbol}=Symbol[],
                        squared_error::Bool=true)

Creates a dataframe from the test dataframe and a supplied prediction.  

The array names supplies the names for the columns, otherwise will generate default names.

Also generates error columns which are the difference between predictions and test data.
If `squared_error`, will also create a column with squared error.

Note that this currently does nothing to handle transformations of the data.
"""
function getTestAnalysisData(dh::AbstractDH, ŷ::Array; names::Vector{Symbol}=Symbol[],
                             squared_error::Bool=true)
    # convert vectors to matrices
    if length(size(ŷ)) == 1
        ŷ = reshape(ŷ, (length(ŷ), 1))
    end
    @assert size(ŷ,2) == length(dh.colsOutput) ("Supplied array must have same number of 
                                                  columns as the handler's output.")
    if length(names) == 0
        names = [Symbol(string(col, "_hat")) for col in dh.colsOutput]
    end
    @assert length(dh.colsOutput) == length(names) ("Wrong number of provided names.")
    # if ŷ is short it is assumed to correspond to begining of dataframe, useful for 
    # time series where only a partial sequence has been generated
    df = dh.dfTest[1:size(ŷ)[1], :]
    for (idx, name) ∈ enumerate(names)
        df[name] = ŷ[:, idx]
        orig_col = convert(Vector, df[dh.colsOutput[idx]])
        err = ŷ[:, idx] - orig_col
        df[Symbol(string(name, "_Error"))] = err
        if squared_error
            df[Symbol(string(name, "_Error²"))] = err.^2
        end
    end
    df
end
export getTestAnalysisData



