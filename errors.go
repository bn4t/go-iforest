package go_iforest

import "errors"

// Errors returned when the arguments passed to NewIForest are invalid.
var (
	ErrNoSamplesProvided       = errors.New("no samples provided")
	ErrTooLargeSubSamplingSize = errors.New("specified sub-sampling size is larger than amount of provided samples")
	ErrInvalidSubSamplingSize  = errors.New("sub-sampling size must be at least 2")
	ErrInvalidTreeCount        = errors.New("number of trees must be at least 1")
	ErrInvalidData             = errors.New("invalid sample data")
)
