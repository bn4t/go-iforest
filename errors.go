package main

import "errors"

var (
	ErrTooLargeSubSamplingSize = errors.New("specified sub-sampling size is larger than amount of provided samples")
	ErrNoSamplesProvided       = errors.New("no samples provided")
)
