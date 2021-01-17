// Package go_iforest is an implementation of isolation forests
// as defined in https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf.

package go_iforest

import (
	"math"
	"math/rand"
)

type IForest struct {
	Trees           []*ITree
	SubSamplingSize int
}

// NewIForest creates a new IForest and trains it on the provided data X
// make sure to call rand.Seed() before calling this function to ensure that
// a sufficiently random sub-sample is chosen.
func NewIForest(X [][]float64, trees int, subSamplingSize int) (*IForest, error) {
	if len(X) == 0 {
		return nil, ErrNoSamplesProvided
	}

	if subSamplingSize > len(X) {
		return nil, ErrTooLargeSubSamplingSize
	}

	forest := IForest{Trees: []*ITree{}, SubSamplingSize: subSamplingSize}

	// always choose a different sub-sample from the dataset
	for i := 0; i < trees; i++ {

		// duplicate the multidimensional X slice to prevent modifying it
		// when selecting hte sub-sample
		duplicateX := make([][]float64, len(X))
		for i := range X {
			duplicateX[i] = make([]float64, len(X[i]))
			copy(duplicateX[i], X[i])
		}

		forest.Trees = append(forest.Trees, NewITree(subSample(duplicateX, subSamplingSize)))
	}

	return &forest, nil
}

// CalculateAnomalyScore calculates an anomaly score based for a sample x
func (f *IForest) CalculateAnomalyScore(x []float64) float64 {
	var sumPathLength float64

	for _, T := range f.Trees {
		sumPathLength += PathLength(x, T.RootNode, 0)
	}

	avgPath := sumPathLength / float64(len(f.Trees))
	return math.Pow(2, -avgPath/avgPathLength(float64(f.SubSamplingSize)))
}

// subSample chooses a random sub sample from the provided data X
func subSample(X [][]float64, v int) [][]float64 {
	var r int
	var sample [][]float64

	for i := 0; i < v; i++ {
		r = rand.Intn(len(X))
		sample = append(sample, X[r])
		X = removeEl(X, r)
	}

	return sample
}

func removeEl(s [][]float64, i int) [][]float64 {
	s[i] = s[len(s)-1]
	// We do not need to put s[i] at the end, as it will be discarded anyway
	return s[:len(s)-1]
}
