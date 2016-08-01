
"""
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
    This constructor makes a fractional train, test split.  
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
    # TODO this was copy pasted, not from here
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


"""
Assigns the training data.  X output will be of shape (samples, seq_length, seq_width).
One should be extremely careful if not sorting.

Note that this is silent if the dataframe is empty.
"""
function assignTrain!(dh::TimeSeriesHandler; sort::Bool=true)
    dh.X_train, dh.y_train = _get_assign_data(:dfTrain, dh, sort=sort)
end
export assignTrain!


"""
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
Assigns both training and testing data.
"""
function assign!(dh::TimeSeriesHandler; sort::Bool=true)
    assignTrain!(dh, sort=sort)
    assignTest!(dh, sort=sort)
    return
end
export assign!


"""
Splits the data by time-index.  All datapoints with τ up to and including the timeindex
given (τ₀) will be in the training set, while all those with τ > τ₀ will be in 
the test set.
"""
function split!(dh::TimeSeriesHandler, τ₀::Integer; assign::Bool=true, sort::Bool=true)
    constr = dh.df[dh.timeindex] .<= τ₀
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
Splits the dataframe by the number of sequences in the test set.  This does nothing to
account for the possibility of missing data.

Note that the actual number of usable test sequences in the resulting test set is of
course greater than n_sequences.
"""
function splitByNSequences!(dh::TimeSeriesHandler, n_sequences::Integer;
                            assign::Bool=true, sort::Bool=true)
    τ_max = maximum(dh.df[:, dh.timeindex])
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
Gets the training X, y pair where X is squashed using getSquahdedTrainMatrix.
If flatten, also flatten y.
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
Uses the supplied prediction function to generate a sequence of a specified length.
The sequence uses the end of the training dataset as initial input.

Note that this only makes sense if the intersection between the input and output columns
isn't ∅.  For now we enforce that they must be identical.
"""
function generateSequence{T}(predict::Function, dh::TimeSeriesHandler{T},
                             seq_length::Integer=8;
                             on_matrix::Bool=false)
    @assert dh.colsInput == dh.colsOutput "Input and output columns much match."
    testsize = (seq_length, length(dh.colsInput))     
    lastX = reshape(convert(Array{T}, dh.dfTrain[(end-dh.seq_length+1):end, dh.colsInput]),
                    (1, dh.seq_length, length(dh.colsInput)))
    if on_matrix
        lastX_reshaped = reshape(lastX, (1, prod(size(lastX))))
    else
        lastX_reshaped = lastX
    end
    yhat = Array{T}(testsize)
    for i in 1:testsize[1]
        yhat[i, :] = predict(lastX_reshaped)
        lastX = circshift(lastX, (0, -1, 0)) 
        lastX[1, end, :] = yhat[i, :]
        if on_matrix
            lastX_reshaped = reshape(lastX, (1, prod(size(lastX))))
        else
            lastX_reshaped = lastX
        end
    end
    return yhat
end
export generateSequence


"""
Uses the supplied prediction function to attempt to predict the entire test set.
Note that this assumes that the test set is ordered, sequential and immediately follows
the training set.

For now we enforce that the input and output columns are identical.
"""
function generateTest(predict::Function, dh::TimeSeriesHandler;
                      on_matrix::Bool=true)
    return generateSequence(predict, dh, size(dh.dfTest)[1], on_matrix=on_matrix)
end
export generateTest


"""
Returns y_test directly from the dataframe for comparison with the output of generateTest.
"""
function getRawTestTarget{T}(dh::TimeSeriesHandler{T})
    return convert(Array{T}, dh.dfTest[:, dh.colsOutput])
end
export getRawTestTarget


