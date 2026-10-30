package go_iforest

import (
	"errors"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// gaussianData returns n samples with the provided amount of features, drawn
// from a standard normal distribution.
func gaussianData(n, features int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	X := make([][]float64, n)
	for i := range X {
		X[i] = make([]float64, features)
		for j := range X[i] {
			X[i][j] = rng.NormFloat64()
		}
	}
	return X
}

func TestNewIForestRejectsInvalidArguments(t *testing.T) {
	valid := [][]float64{{1, 2}, {2, 3}, {3, 4}}

	tests := []struct {
		name            string
		X               [][]float64
		trees           int
		subSamplingSize int
		want            error
	}{
		{"no samples", nil, 10, 10, ErrNoSamplesProvided},
		{"no trees", valid, 0, 2, ErrInvalidTreeCount},
		{"negative trees", valid, -1, 2, ErrInvalidTreeCount},
		{"zero sub-sampling size", valid, 10, 0, ErrInvalidSubSamplingSize},
		{"sub-sampling size of one", valid, 10, 1, ErrInvalidSubSamplingSize},
		{"sub-sampling size too large", valid, 10, 4, ErrTooLargeSubSamplingSize},
		{"no features", [][]float64{{}, {}}, 10, 2, ErrInvalidData},
		{"ragged samples", [][]float64{{1, 2}, {3}}, 10, 2, ErrInvalidData},
		{"NaN feature", [][]float64{{1, 2}, {math.NaN(), 3}}, 10, 2, ErrInvalidData},
		{"infinite feature", [][]float64{{1, 2}, {3, math.Inf(1)}}, 10, 2, ErrInvalidData},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			forest, err := NewIForest(tt.X, tt.trees, tt.subSamplingSize)
			if !errors.Is(err, tt.want) {
				t.Fatalf("NewIForest() error = %v, want %v", err, tt.want)
			}
			if forest != nil {
				t.Fatalf("NewIForest() = %v, want nil forest", forest)
			}
		})
	}
}

func TestCalculateAnomalyScoreOnNormalData(t *testing.T) {
	rand.Seed(1)
	X := gaussianData(1000, 4, 1)

	forest, err := NewIForest(X, 100, 256)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}
	if len(forest.Trees) != 100 {
		t.Fatalf("len(Trees) = %d, want 100", len(forest.Trees))
	}
	if forest.SubSamplingSize != 256 {
		t.Fatalf("SubSamplingSize = %d, want 256", forest.SubSamplingSize)
	}

	var sum float64
	for _, sample := range X {
		score := forest.CalculateAnomalyScore(sample)
		if math.IsNaN(score) || score <= 0 || score >= 1 {
			t.Fatalf("CalculateAnomalyScore(%v) = %v, want a score between 0 and 1", sample, score)
		}
		sum += score
	}

	// On normally distributed data the average score should be close to 0.5.
	if mean := sum / float64(len(X)); mean < 0.43 || mean > 0.57 {
		t.Errorf("mean anomaly score = %v, want approximately 0.5", mean)
	}
}

func TestAnomaliesRankHighest(t *testing.T) {
	rand.Seed(2)
	X := gaussianData(1000, 4, 2)

	forest, err := NewIForest(X, 100, 256)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	maxNormal := 0.0
	for _, sample := range X {
		maxNormal = math.Max(maxNormal, forest.CalculateAnomalyScore(sample))
	}

	outliers := [][]float64{
		{8, 8, 8, 8},
		{-8, -8, -8, -8},
		{8, -8, 8, -8},
	}
	for _, outlier := range outliers {
		if score := forest.CalculateAnomalyScore(outlier); score <= maxNormal {
			t.Errorf("outlier %v scored %v, want more than the highest normal score %v", outlier, score, maxNormal)
		}
	}
}

func TestIdenticalSamplesScoreHalf(t *testing.T) {
	X := make([][]float64, 100)
	for i := range X {
		X[i] = []float64{5, 5}
	}

	rand.Seed(3)
	forest, err := NewIForest(X, 50, 64)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	// Every sample is indistinguishable from every other sample, so it is
	// maximally normal.
	if score := forest.CalculateAnomalyScore([]float64{5, 5}); math.Abs(score-0.5) > 1e-12 {
		t.Errorf("anomaly score = %v, want 0.5", score)
	}
}

func TestConstantFeatureIsIgnored(t *testing.T) {
	rand.Seed(4)
	X := gaussianData(200, 2, 4)
	for i := range X {
		X[i] = append(X[i], 42) // third feature never varies
	}

	forest, err := NewIForest(X, 100, 128)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	// A constant feature cannot separate samples; training must still work and
	// produce meaningful scores.
	if score := forest.CalculateAnomalyScore([]float64{8, 8, 42}); score < 0.6 {
		t.Errorf("anomaly score = %v, want an anomaly score above 0.6", score)
	}
}

func TestNegativeFeatureValues(t *testing.T) {
	// Regression test: the split point used to be based on a maximum that was
	// initialized to math.SmallestNonzeroFloat64, which produced degenerate
	// splits for data with only negative values.
	rand.Seed(5)
	X := make([][]float64, 500)
	for i := range X {
		X[i] = []float64{-10 - rand.Float64(), -10 - rand.Float64()}
	}
	outlier := []float64{-100, -100}
	X = append(X, outlier)

	forest, err := NewIForest(X, 100, 256)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	var sum float64
	for _, sample := range X[:len(X)-1] {
		score := forest.CalculateAnomalyScore(sample)
		if score < 0 || score > 1 {
			t.Fatalf("score = %v, want a score between 0 and 1", score)
		}
		sum += score
	}

	if mean := sum / float64(len(X)-1); mean < 0.4 || mean > 0.55 {
		t.Errorf("mean anomaly score = %v, want approximately 0.45", mean)
	}
	if outlierScore, normalScore := forest.CalculateAnomalyScore(outlier), forest.CalculateAnomalyScore(X[0]); outlierScore <= normalScore {
		t.Errorf("outlier score = %v, want more than normal score %v", outlierScore, normalScore)
	}
}

func TestTwoSamplesPerTree(t *testing.T) {
	// With two samples per tree, a tree can either split them (path length 1)
	// or cannot split them at all (also path length c(2) = 1). Every score is
	// therefore exactly 0.5.
	rand.Seed(6)
	X := [][]float64{{0, 0}, {1, 1}, {2, 2}, {3, 3}}

	forest, err := NewIForest(X, 100, 2)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	for _, sample := range X {
		if score := forest.CalculateAnomalyScore(sample); math.Abs(score-0.5) > 1e-12 {
			t.Errorf("CalculateAnomalyScore(%v) = %v, want 0.5", sample, score)
		}
	}
}

func TestCalculateAnomalyScoreRejectsInvalidSamples(t *testing.T) {
	rand.Seed(7)
	forest, err := NewIForest(gaussianData(100, 3, 7), 10, 64)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	samples := [][]float64{
		nil,
		{1, 2},
		{1, 2, 3, 4},
		{1, math.NaN(), 3},
		{1, math.Inf(1), 3},
	}
	for _, sample := range samples {
		if score := forest.CalculateAnomalyScore(sample); !math.IsNaN(score) {
			t.Errorf("CalculateAnomalyScore(%v) = %v, want NaN", sample, score)
		}
	}
}

func TestCalculateAnomalyScoreWithoutTrees(t *testing.T) {
	var forest *IForest
	if score := forest.CalculateAnomalyScore([]float64{1, 2}); !math.IsNaN(score) {
		t.Errorf("nil forest score = %v, want NaN", score)
	}

	if score := (&IForest{SubSamplingSize: 10}).CalculateAnomalyScore([]float64{1, 2}); !math.IsNaN(score) {
		t.Errorf("empty forest score = %v, want NaN", score)
	}

	withNilTree := &IForest{Trees: []*ITree{nil}, SubSamplingSize: 10}
	if score := withNilTree.CalculateAnomalyScore([]float64{1, 2}); !math.IsNaN(score) {
		t.Errorf("forest with nil tree score = %v, want NaN", score)
	}
}

func TestNewIForestDoesNotModifyInput(t *testing.T) {
	X := gaussianData(100, 3, 12)
	original := make([][]float64, len(X))
	for i := range X {
		original[i] = append([]float64(nil), X[i]...)
	}

	rand.Seed(12)
	if _, err := NewIForest(X, 20, 64); err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	for i := range X {
		for j := range X[i] {
			if X[i][j] != original[i][j] {
				t.Fatalf("input changed at [%d][%d]: %v, want %v", i, j, X[i][j], original[i][j])
			}
		}
	}
}

func TestAnomalyScoresAreAffineInvariant(t *testing.T) {
	// Isolation forest splits are based on ranges, so scaling or shifting a
	// feature by a positive factor must not change any score.
	X := gaussianData(300, 3, 11)
	scales := []float64{0.01, 3, 1000}
	offsets := []float64{-1000, 0, 42}

	transformed := make([][]float64, len(X))
	for i := range X {
		transformed[i] = make([]float64, len(X[i]))
		for j := range X[i] {
			transformed[i][j] = X[i][j]*scales[j] + offsets[j]
		}
	}

	rand.Seed(11)
	forest, err := NewIForest(X, 50, 128)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	rand.Seed(11)
	transformedForest, err := NewIForest(transformed, 50, 128)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	for i := range X {
		want := forest.CalculateAnomalyScore(X[i])
		got := transformedForest.CalculateAnomalyScore(transformed[i])
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("score changed from %v to %v after positive affine transformation", want, got)
		}
	}
}

func TestNewIForestIsDeterministicWithSeed(t *testing.T) {
	X := gaussianData(300, 3, 8)

	rand.Seed(100)
	first, err := NewIForest(X, 20, 128)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	rand.Seed(100)
	second, err := NewIForest(X, 20, 128)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	for _, sample := range X[:20] {
		if firstScore, secondScore := first.CalculateAnomalyScore(sample), second.CalculateAnomalyScore(sample); firstScore != secondScore {
			t.Fatalf("scores differ with identical seed: %v != %v", firstScore, secondScore)
		}
	}
}

func TestCalculateAnomalyScoreIsConcurrentSafe(t *testing.T) {
	rand.Seed(9)
	X := gaussianData(200, 2, 9)
	forest, err := NewIForest(X, 20, 128)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if score := forest.CalculateAnomalyScore(X[j%len(X)]); math.IsNaN(score) {
					t.Errorf("concurrent score = NaN")
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestExtremeFeatureValues(t *testing.T) {
	// max-min overflows float64 for these values; the split point must still
	// stay finite so that the tree remains usable.
	X := [][]float64{
		{-math.MaxFloat64, 0},
		{math.MaxFloat64, 1},
		{0, 2},
		{1, 3},
	}

	rand.Seed(13)
	forest, err := NewIForest(X, 100, 4)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	for _, sample := range X {
		if score := forest.CalculateAnomalyScore(sample); math.IsNaN(score) || score <= 0 || score >= 1 {
			t.Errorf("CalculateAnomalyScore(%v) = %v, want a score between 0 and 1", sample, score)
		}
	}
}

func TestAdjacentFloatValues(t *testing.T) {
	// Adjacent float64 values leave almost no room for a split point, so it
	// often rounds to the minimum. Such nodes must become leaves instead of
	// creating empty branches.
	next := math.Nextafter(1, math.Inf(1))
	X := [][]float64{{1}, {next}, {1}, {next}}

	rand.Seed(14)
	forest, err := NewIForest(X, 100, 4)
	if err != nil {
		t.Fatalf("NewIForest() error = %v", err)
	}

	for _, sample := range X {
		if score := forest.CalculateAnomalyScore(sample); math.IsNaN(score) || score <= 0 || score >= 1 {
			t.Errorf("CalculateAnomalyScore(%v) = %v, want a score between 0 and 1", sample, score)
		}
	}
}

func TestITreePathLength(t *testing.T) {
	rand.Seed(10)
	tree := NewITree([][]float64{{0, 0}, {1, 1}})

	for _, sample := range [][]float64{{0, 0}, {1, 1}} {
		if got := tree.PathLength(sample); got != 1 {
			t.Errorf("PathLength(%v) = %v, want 1", sample, got)
		}
	}

	if got := tree.PathLength([]float64{0}); !math.IsNaN(got) {
		t.Errorf("PathLength() with wrong dimensions = %v, want NaN", got)
	}

	if got := PathLength(nil, nil, 3); got != 3 {
		t.Errorf("PathLength(nil, nil, 3) = %v, want 3", got)
	}
}

func TestNewITreeHandlesEmptyInput(t *testing.T) {
	tests := []struct {
		name string
		X    [][]float64
		want float64
	}{
		{"nil", nil, 0},
		{"empty", [][]float64{}, 0},
		{"no features", [][]float64{{}, {}}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tree := NewITree(tt.X)
			if tree == nil || tree.rootNode == nil {
				t.Fatal("NewITree() returned a tree without a root node")
			}
			if got := tree.PathLength(nil); got != tt.want {
				t.Errorf("PathLength(nil) = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAvgPathLength(t *testing.T) {
	tests := map[float64]float64{
		0:   0,
		1:   0,
		2:   1,
		3:   1.207392357586557,
		256: 10.244770920116851,
	}
	for n, want := range tests {
		if got := avgPathLength(n); math.Abs(got-want) > 1e-12 {
			t.Errorf("avgPathLength(%v) = %v, want %v", n, got, want)
		}
	}

	previous := avgPathLength(2)
	for n := 3.0; n <= 1000; n++ {
		got := avgPathLength(n)
		if got <= previous {
			t.Fatalf("avgPathLength(%v) = %v, want more than avgPathLength(%v) = %v", n, got, n-1, previous)
		}
		previous = got
	}
}
