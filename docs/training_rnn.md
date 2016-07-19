# Some Advice on Training RNN's

Training an RNN can be rather tricky, training one for sequence prediction can be especially tricky because there is so little literature on it (oddly).  Here
are a couple of things that I have noticed that could be very useful for quick training.

+ It's essential to rescale and center the input.  This really seems far too obvious to write down, but sometimes I'm wont to do stupid things while testing or
  just generally screwing around.  Sometimes the training is also very sensitive to the rescaling.  Probably better to rescale to the full range rather than
  just using the standard deviation.
+ The result can be extremely sensitive to batch size, depending on how you train.  Certain types of input processing (in particular, `tflearn`) will run over
  all of the data in batches (all data is one epoch), so the number of steps in an epoch will be inversely related to the batch size.  You can't have too few
  steps in an epoch or else you'll only get a few gradients in each epoch, it's not clear to me whether this would even work with a huge number of epochs
  (certainly it would take forever).
+ Obviously learning rate is crucial, but also be very careful about which algorithm you use.  While screwing around I noticed that the optimizers are much more
  prone to "popping out" of the vicinity of the minima (than in the non-recurrent case).  If you get too much popping out, turn the learning rate *way* down and
  gradually bring it back up.
+ Despite this "popping out" behavior, methods with momentum don't always seem to be worse.
+ Shuffling data can sometimes be helpful, sometimes ordered data can lead the search in odd directions.
+ Deeply recurrent neural networks (i.e. multiple recurrent layers) can be a fucking nightmare.  Again, you'll have to be very careful with the learning rate,
  batch size and optimization method used.  They typically require much smaller learning rates.  See if you can achieve whatever it is you are attempting by
  adding ordinary fully-connected layers instead.  

