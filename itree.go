package main

import (
	"math"
	"math/rand"
)

const EulersConstant = 0.5772156649

type ITree struct {
	// The root node of the tree
	// From this node all child nodes can be accessed by
	// following the Node{Left,Right} variables
	RootNode *Node

	// The height limit of the tree, calculated using
	// ceiling(log2(v)) where v is the sub-sampling size
	HeightLimit float64

	// The average path length of this tree, calculated
	// using avgPathLength()
	AvgPathLength float64
}

type Node struct {
	// size of the remaining samples if this is an external node.
	// This is relevant if the max tree size is reached
	size int

	// the split point of this node
	SplitPoint float64

	// index of the attribute used for the split point
	SplitAttrIndex int

	NodeLeft  *Node
	NodeRight *Node
	External  bool
}

func NewITree(X [][]float64) *ITree {
	l := math.Ceil(math.Log2(float64(len(X))))
	var indices []int

	// get initial indices
	for k := range X {
		indices = append(indices, k)
	}

	return &ITree{
		RootNode:      NextNode(X, indices, 0, l),
		HeightLimit:   l,
		AvgPathLength: avgPathLength(float64(len(X))),
	}
}

// NextNode creates a new node in the tree
func NextNode(X [][]float64, indices []int, e float64, l float64) *Node {
	// return an external node, if only one sample remains
	// or the max tree height is reached.
	if e >= l || len(indices) <= 1 {
		return &Node{
			size:     len(indices),
			External: true,
		}
	}

	// select a random attribute q
	q := rand.Intn(len(X[0]))

	// choose a split point p between the max and min value of the attribute
	p := selectSplitPoint(X, indices, q)

	var IndicesL []int
	var IndicesR []int

	// split up the samples in X at the chosen split point
	for _, v := range indices {
		if X[v][q] < p {
			IndicesL = append(IndicesL, v)
		} else {
			IndicesR = append(IndicesR, v)
		}
	}

	return &Node{
		SplitPoint:     p,
		SplitAttrIndex: q,
		NodeLeft:       NextNode(X, IndicesL, e+1, l),
		NodeRight:      NextNode(X, IndicesR, e+1, l),
		External:       false,
	}
}

// PathLength derives the path length for an instance x using a Tree T
// e is the current path length (0 on the first call to this method)
func PathLength(x []float64, T *Node, e int) float64 {

	if T.External {
		if T.size <= 1 {
			return float64(e)
		} else {
			// return size plus an adjustment of c(size) where size is the size of the remaining
			// samples when this external node was created (either because the tree size limit was reached,
			// or because only a single sample was remaining -> size = 1).
			// c is the avgPathLength method
			return float64(e) + avgPathLength(float64(T.size))
		}
	}

	if x[T.SplitAttrIndex] < T.SplitPoint {
		return PathLength(x, T.NodeLeft, e+1)
	} else {
		return PathLength(x, T.NodeRight, e+1)
	}
}

// selectSplitPoint chooses a random split point between the max and min value
// of attribute attrIndex from the provided indices of X
func selectSplitPoint(X [][]float64, indices []int, attrIndex int) float64 {
	max := math.SmallestNonzeroFloat64
	min := math.MaxFloat64

	for _, v := range indices {
		if X[v][attrIndex] > max {
			max = X[v][attrIndex]
		}
		if X[v][attrIndex] < min {
			min = X[v][attrIndex]
		}
	}

	// calculate a random float in the range between min and max
	return min + rand.Float64()*(max-min)
}

func avgPathLength(n float64) float64 {
	return 2*(math.Log(n-1)+EulersConstant) - ((2 * (n - 1)) / n)
}
