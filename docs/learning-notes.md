# Learning Notes

## 1. Sequence-aware validation split

I noticed that GTSRB training images contain frame sequences. Images from the
same sequence are very similar. A random image split can place nearby frames in
both the training set and the validation set. This makes the validation result
too optimistic.

I changed the validation loader to group samples by `(class_id, sequence_id)`.
Each sequence now belongs to only one split. This gives a more realistic
validation result.
