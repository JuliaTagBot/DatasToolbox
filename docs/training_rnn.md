# Some Advice on Training RNN's

Training an RNN can be rather tricky, training one for sequence prediction can be especially tricky because there is so little literature on it (oddly).  Here
are a couple of things that I have noticed that could be very useful for quick training.

+ It's essential to rescale and center the input.  This really seems far too obvious to write down, but sometimes I'm wont to do stupid things while testing or
  just generally screwing around.
+ The result can be extremely sensitive to batch size, depending on how you train.  Certain types of input processing (in particular, `tflearn`) will run over
  all of the data in batches (all data is one epoch), so the number of steps in an epoch will be inversely related to the batch size.  You can't have too few
  steps in an epoch or else you'll only get a few gradients in each epoch, it's not clear to me whether this would even work with a huge number of epochs
  (certainly it would take forever).
+ Obviously learning rate is crucial, but also be very careful about which algorithm you use.  While screwing around I noticed that the optimizers are much more
  prone to "popping out" of the vicinity of the minima (than in the non-recurrent case) so methods with momentum like ADAM tend to be problematic.

