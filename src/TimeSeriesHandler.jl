
"""
    TimeSeriesHandler{T} <: AbstractDH{T}

Type for handling time series data.  As with DataHandler it is intended taht most of
the reformatting of the dataframe is done before passing it to an instance of this type.

The parameter T specifies the datatype of the input, output arrays.
"""
type TimeSeriesHandler{T} <: AbstractDH{T}

    # dataframe where all data is kept
    df::DataFrame

    colsInput::Array{Symbol, 1}
    colsOutput::Array{Symbol, 1}

    colsNormalize::Array{Symbol, 1}

    mu::Array{T, 1}
    norm::Array{T, 1}
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

    # the symbol of the timeindex
    timeindex::Symbol
    # the length of sequences (only fixed for now)
    seq_length::Int64

    """
        TimeSeriesHandler(df::DataFrame, timeindex::Symbol, seq_length::Integer;
                          shuffle::Bool=false, n_test_sequences::Integer=0,
                          input_cols::Array{Symbol}=Symbol[],
                          output_cols::Array{Symbol}=Symbol[],
                          normalize_cols::Array{Symbol}=Symbol[],
                          assign::Bool=false, userange::Bool=false)

    Creates an object converting the data contained in `df` to properly formatted
    time series data with fixed-length sequences.  It is required that the input dataframe
    have a time index, the name of which should be passed as the second argument.
    The keyword arguments are otherwise the same as for `DataHandler` but note that in 
    this case it makes sense for `input_cols == output_cols` (in fact that is probably the
    most common use case).
    """
    function TimeSeriesHandler(df::DataFrame, timeindex::Symbol, seq_length::Integer; 
                               shuffle::Bool=false,
                               n_test_sequences::Integer=0,
                               input_cols::Array{Symbol}=Symbol[], 
                               output_cols::Array{Symbol}=Symbol[],
                               normalize_cols::Array{Symbol}=Symbol[],
                               assign::Bool=false,
                               userange::Bool=false)
        ndf = copy(df)
        o = new(ndf, input_cols, output_cols, normalize_cols)
        o.userange = userange
        o.timeindex = timeindex
        o.seq_length = seq_length
        splitByNSequences!(o, n_test_sequences, assign=assign)
        computeNormalizeParameters!(o, dataset=:dfTrain)
        if canNormalize(o)
            normalizeTrain!(o)
            if size(o.dfTest)[1] > 0 normalizeTest!(o) end
        end
        return o
    end
end
export TimeSeriesHandler


"""
Used by assignTrain!, and assignTest!
"""
function _get_assign_data{T}(dataframe::Symbol, dh::TimeSeriesHandler{T}; sort::Bool=true)
    df = getfield(dh, dataframe)
    if isempty(df) return end
    if sort sort!(df, cols=[dh.timeindex]) end 
    npoints = size(df)[1] - dh.seq_length
    X = Array{T}(npoints, dh.seq_length, length(dh.colsInput))
    y = Array{T}(npoints, length(dh.colsOutput))
    # the loop can only go this far
    for i in 1:npoints
        nextx = reshape(convert(Array, df[i:(i+dh.seq_length-1), dh.colsInput]),
                        (1, dh.seq_length, length(dh.colsInput)))
        X[i, :, :] = nextx
        y[i, :] = convert(Array, df[i+dh.seq_length, dh.colsOutput])
    end
    return X, y
end


# TODO: this is not working, need to use shared arrays, may not be able to use @parallel
"""
Parallel version, used by assignTrain! and assignTest!.
"""
function _get_assign_data_parallel{T}(dataframe::Symbol, dh::TimeSeriesHandler{T}; 
                                      sort::Bool=true)
    df = getfield(dh, dataframe)
    if isempty(df) return end
    if sort sort!(df, cols=[dh.timeindex]) end 
    npoints = size(df)[1] - dh.seq_length
    X = Array{T}(npoints, dh.seq_length, length(dh.colsInput))
    y = Array{T}(npoints, length(dh.colsOutput))
    # this SHOULD work
    @parallel for i in 1:npoints
        nextx = reshape(convert(Array, df[i:(i+dh.seq_length-1), dh.colsInput]),
                        (1, dh.seq_length, length(dh.colsInput)))
        X[i, :, :] = nextx
        y[i, :] = convert(Array, df[i+dh.seq_length, dh.colsOutput])
    end
    return X, y
end


"""
    assignTrain!(dh::TimeSeriesHandler; sort::Bool=true, parallel::Bool=false)

Assigns the training data.  X output will be of shape (samples, seq_length, seq_width).
If `sort` is true, will sort the dataframe first.  One should be extremely careful if
`sort` is false.  If `parallel` is true the data will be generated in parallel (using 
workers, not threads).  This is useful because this data manipulation is complicated and
potentially slow.

Note that this is silent if the dataframe is empty.

**TODO** I'm pretty sure the parallel version isn't working right because it doesn't
use shared arrays.  Revisit in v0.5 with threads.
"""
function assignTrain!(dh::TimeSeriesHandler; sort::Bool=true, parallel::Bool=false)
    # TODO, again parallel not currently working
    if parallel
        gad = _get_assign_data_parallel
    else
        gad = _get_assign_data
    end
    dh.X_train, dh.y_train = gad(:dfTrain, dh, sort=sort)
end
export assignTrain!


"""
    assignTest!(dh::TimeSeriesHandler; sort::Bool=true)

Assigns the test data.  X output will be of shape (samples, seq_length, seq_width).
One should be extremely careful if not sorting.

Note that in the time series case this isn't very useful.  One should instead use
one of the assigned prediction functions.

Note that this is silent if the dataframe is empty.
"""
function assignTest!(dh::TimeSeriesHandler; sort::Bool=true)
    dh.X_test, dh.y_test = _get_assign_data(:dfTest, dh, sort=sort)
end
export assignTest!


"""
    assign!(dh::TimeSeriesHandler; sort::Bool=true)

Assigns both training and testing data for the `TimeSeriesHandler`.
"""
function assign!(dh::TimeSeriesHandler; sort::Bool=true)
    assignTrain!(dh, sort=sort)
    assignTest!(dh, sort=sort)
    return
end
export assign!


"""
    split!(dh::TimeSeriesHandler, τ₀::Integer; assign::Bool=true, sort::Bool=true)

Splits the data by time-index.  All datapoints with τ up to and including the timeindex
given (τ₀) will be in the training set, while all those with τ > τ₀ will be in 
the test set.
"""
function split!(dh::TimeSeriesHandler, τ₀::Integer; assign::Bool=true, sort::Bool=true)
    constr = dropnull(dh.df[dh.timeindex]) .≤ τ₀
    dfTrain = dh.df[constr, :]
    dfTest = dh.df[!constr, :]
    if !isempty(dfTrain)
        dh.dfTrain = dfTrain
    end
    if !isempty(dfTest)
        dh.dfTest = dfTest
    end
    if assign assign!(dh, sort=sort) end
    return
end
export split!


"""
    splitByNSequences!(dh::TimeSeriesHandler, n_sequences::Integer;
                       assign::Bool=true, sort::Bool=true)

Splits the dataframe by the number of sequences in the test set.  This does nothing to
account for the possibility of missing data.

Note that the actual number of usable test sequences in the resulting test set is of
course greater than n_sequences.
"""
function splitByNSequences!(dh::TimeSeriesHandler, n_sequences::Integer;
                            assign::Bool=true, sort::Bool=true)
    τ_max = maximum(dropnull(dh.df[:, dh.timeindex]))
    nτ = n_sequences*dh.seq_length
    τ₀ = τ_max - nτ
    split!(dh, τ₀, assign=assign, sort=sort)
    return
end
export splitByNSequences!


"""
Squashes the last two indices of a rank-3 tensor into a matrix.  For internal use.
"""
function _squashXTensor(X::Array, dh::TimeSeriesHandler)
    return reshape(X, (size(X)[1], dh.seq_length*length(dh.colsInput))) 
end


"""
    getSquashedTrainMatrix(dh::TimeSeriesHandler)

Gets a training input tensor in which all the inputs are arranged along a single axis
(i.e. in a matrix).

Assumes the handler's X_train is defined.
"""
function getSquashedTrainMatrix(dh::TimeSeriesHandler)
    if !isdefined(dh, :X_train)
        error("Training input not defined.  Should call assign!.")
    end
    return _squashXTensor(dh.X_train, dh)
end
export getSquashedTrainMatrix


"""
    getSquashedTestMatrix(dh::TimeSeriesHandler)

Gets a test input tensor in which all the inputs are arranged along a single axis
(i.e. in a matrix).

Assumes the handler's X_test is defined.
"""
function getSquashedTestMatrix(dh::TimeSeriesHandler)
    if !isdefined(dh, :X_test)
        error("Test input not defined.  Should call assign!.")
    end
    return _squashXTensor(dh.X_test, dh)
end
export getSquashedTestMatrix


"""
    getSquashedTrainData(dh::TimeSeriesHandler; flatten::Bool=false)

Gets the training X, y pair where X is squashed using `getSquahdedTrainMatrix`.
If `flatten`, also flatten `y`.
"""
function getSquashedTrainData(dh::TimeSeriesHandler; flatten::Bool=false)
    if flatten
        @assert size(dh.y_train)[2] == 1 "Attempting to flatten mis-shaped matrix."
        y_train = squeeze(dh.y_train, 2)
    else
        y_train = dh.y_train
    end
    return getSquashedTrainMatrix(dh), y_train
end
export getSquashedTrainData


"""
    trainOnMatrix(train::Function, dh::TimeSeriesHandler; flatten::Bool=true)

Trains an object designed to take a matrix (as opposed to a rank-3 tensor) as input.
The first argument is the function used to train.

If flatten, the training output is converted to a vector.  Most methods that take matrix
input take vector target data.

Assumes the function takes input of the form train(X, y).
"""
function trainOnMatrix(train::Function, dh::TimeSeriesHandler; flatten::Bool=true)
    # turn off flatten if y_train is a matrix
    if length(dh.colsOutput) > 1
        flatten = false
    end
    X = getSquashedTrainMatrix(dh)
    if flatten
        y_train = squeeze(dh.y_train, 2)
    else
        y_train = dh.y_train
    end
    train(X, y_train)
end
export trainOnMatrix


"""
    _check_and_reshape(lastX, on_matrix)

Private method used by generateSequence.
"""
function _check_and_reshape(lastX::Array, on_matrix::Bool)
    if on_matrix
        lastX_reshaped = reshape(lastX, (1, prod(size(lastX))))
    else
        lastX_reshaped = lastX
    end
    lastX_reshaped
end


"""
    _get_next_X(indim, outin_idx, yhat, newcols_funcs)

Private method used by `generateSequence`.
"""
function _get_next_X!{T}(lastX::Array, yhat::Array{T}, outin_idx::Vector)
    for (i, idx) ∈ enumerate(outin_idx)
        lastX[1, end, idx] = yhat[i]
    end
end

function _get_next_X!{T}(lastX::Array, yhat::Array{T}, outin_idx::Vector, newcol_funcs::Dict)
    _get_next_X!(lastX, yhat, outin_idx)
    for (n, f) ∈ newcol_funcs
        lastX[1, end, n] = f(lastX[1, end-1, :])
    end
end

function _get_next_X!{T}(lastX::Array, yhat::Array{T}, outin_idx::Vector, 
                         newcol_func::Function)
    _get_next_X!(lastX, yhat, outin_idx)
    dict = newcol_func(lastX[1, end-1, :])
    for (n, x) ∈ dict
        lastX[1, end, n] = x
    end
end


"""
    generateSequence(predict, dh, seq_length[, newcol_func; on_matrix=false])
                    
Uses the supplied prediction function `predict` to generate a sequence of length `seq_length`.
The sequence uses the end of the training dataset as initial input.

Note that this only makes sense when the output columns are a subset of the input columns.


If a function returning a dictionary or a dictionary of functions `newcol_func` is supplied,
every time a new row of the input is generated, it will have columns specified by 
`newcol_func`.  The dictionary should have keys equal to the column numbers of columns
in the input matrix and values equal to functions that take a `Vector` (the previous
input row) and output a new value for the column number given by the key.  The column
numbers correspond to the index of the column in the specified input columns.

If `on_matrix` is true, the prediction function will take a matrix as input rather than
a rank-3 tensor.
"""
# we keep this method separate for a very tiny amount of efficiency
function generateSequence{T}(predict::Function, dh::TimeSeriesHandler{T},
                             seq_length::Integer;
                             on_matrix::Bool=false)
    outin_idx = indexin(dh.colsOutput, dh.colsInput)
    testsize = (seq_length, length(dh.colsInput))     
    lastX = reshape(convert(Array{T}, dh.dfTrain[(end-dh.seq_length+1):end, dh.colsInput]),
                    (1, dh.seq_length, length(dh.colsInput)))
    lastX_reshaped = _check_and_reshape(lastX, on_matrix)
    yhat = Array{T}(testsize)
    for i in 1:testsize[1]
        pred = predict(lastX_reshaped)
        lastX = circshift(lastX, (0, -1, 0)) 
        _get_next_X!(lastX, pred, outin_idx)
        yhat[i, :] = lastX[1, end, :]
        lastX_reshaped = _check_and_reshape(lastX, on_matrix)
    end
    yhat
end

function generateSequence{T}(predict::Function, dh::TimeSeriesHandler{T},
                             seq_length::Integer, newcol_funcs::Union{Dict,Function};
                             on_matrix::Bool=false)
    outin_idx = indexin(dh.colsOutput, dh.colsInput)
    testsize = (seq_length, length(dh.colsInput))
    lastX = reshape(convert(Array{T}, dh.dfTrain[(end-dh.seq_length+1):end, dh.colsInput]),
                    (1, dh.seq_length, length(dh.colsInput)))
    lastX_reshaped = _check_and_reshape(lastX, on_matrix)
    yhat = Array{T}(testsize)
    for i ∈ 1:testsize[1]
        pred = predict(lastX_reshaped)
        lastX = circshift(lastX, (0, -1, 0))
        _get_next_X!(lastX, pred, outin_idx, newcol_funcs)
        yhat[i, :] = lastX[1, end, :]
        lastX_reshaped = _check_and_reshape(lastX, on_matrix)
    end
    yhat
end
export generateSequence


"""
    generateTest(predict::Function, dh::TimeSeriesHandler; on_matrix::Bool=true)

Uses the supplied prediction function to attempt to predict the entire test set.
Note that this assumes that the test set is ordered, sequential and immediately follows
the training set.

See the documentation for `generateSequence`.
"""
function generateTest(predict::Function, dh::TimeSeriesHandler;
                      on_matrix::Bool=true)
    return generateSequence(predict, dh, size(dh.dfTest)[1], on_matrix=on_matrix)
end
export generateTest


"""
    getRawTestTarget(dh::TimeSeriesHandler)

Returns `y_test` directly from the dataframe for comparison with the output of generateTest.
"""
function getRawTestTarget{T}(dh::TimeSeriesHandler{T})
    return convert(Array{T}, dh.dfTest[:, dh.colsOutput])
end

function getRawTestTarget{T}(dh::TimeSeriesHandler{T}, limit::Integer)
    return convert(Array{T}, dh.dfTest[1:limit, dh.colsOutput])
end

function getRawTestTarget{T}(dh::TimeSeriesHandler{T}, slice::UnitRange)
    return convert(Array{T}, dh.dfTest[slice, dh.colsOutput])
end
export getRawTestTarget


