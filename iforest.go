// Package go_iforest is an implementation of isolation forests
// as defined in https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf.
package go_iforest

import (
	"fmt"
	"math"
	"math/rand"
)

// IForest is an ensemble of isolation trees.
type IForest struct {
	Trees           []*ITree
	SubSamplingSize int
}

// NewIForest creates a new IForest and trains it on the provided data X.
// Make sure to call rand.Seed() before calling this function to ensure that
// a sufficiently random sub-sample is chosen.
func NewIForest(X [][]float64, trees int, subSamplingSize int) (*IForest, error) {
	if len(X) == 0 {
		return nil, ErrNoSamplesProvided
	}

	if trees < 1 {
		return nil, ErrInvalidTreeCount
	}

	if subSamplingSize < 2 {
		return nil, ErrInvalidSubSamplingSize
	}

	if subSamplingSize > len(X) {
		return nil, ErrTooLargeSubSamplingSize
	}

	numFeatures := len(X[0])
	if numFeatures == 0 {
		return nil, fmt.Errorf("%w: samples need at least one feature", ErrInvalidData)
	}

	for i, sample := range X {
		if len(sample) != numFeatures {
			return nil, fmt.Errorf("%w: sample %d has %d features, want %d", ErrInvalidData, i, len(sample), numFeatures)
		}

		for j, value := range sample {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("%w: sample %d feature %d is %v", ErrInvalidData, i, j, value)
			}
		}
	}

	forest := IForest{
		Trees:           make([]*ITree, 0, trees),
		SubSamplingSize: subSamplingSize,
	}

	// Reuse a single index permutation across all trees. A partial
	// Fisher-Yates shuffle draws a uniformly random sub-sample without
	// copying the whole data set once per tree.
	indices := make([]int, len(X))
	for i := range indices {
		indices[i] = i
	}

	for i := 0; i < trees; i++ {
		sample := make([][]float64, subSamplingSize)
		for j := 0; j < subSamplingSize; j++ {
			k := j + rand.Intn(len(indices)-j)
			indices[j], indices[k] = indices[k], indices[j]
			sample[j] = X[indices[j]]
		}

		forest.Trees = append(forest.Trees, NewITree(sample))
	}

	return &forest, nil
}

// CalculateAnomalyScore calculates an anomaly score for a sample x.
// Scores close to 1 indicate anomalies, scores around 0.5 are considered
// normal and scores below 0.5 indicate dense, normal regions.
// It returns NaN if the forest has no trees, if x does not have the same
// amount of features as the training data, or if x contains NaN or Inf.
func (f *IForest) CalculateAnomalyScore(x []float64) float64 {
	if f == nil || len(f.Trees) == 0 || f.SubSamplingSize < 2 {
		return math.NaN()
	}

	for _, value := range x {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return math.NaN()
		}
	}

	var sumPathLength float64
	for _, T := range f.Trees {
		if T == nil {
			return math.NaN()
		}
		sumPathLength += T.PathLength(x)
	}

	avgPath := sumPathLength / float64(len(f.Trees))
	return math.Pow(2, -avgPath/avgPathLength(float64(f.SubSamplingSize)))
}
