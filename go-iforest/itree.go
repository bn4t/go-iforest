package go_iforest

import (
	"math"
	"math/rand"
)

const EulersConstant = 0.577215665

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

	// the splitpoint of this node
	SplitPoint float64

	// index of the attribute used for the splitpoint
	SplitAttrIndex int

	NodeLeft  *Node
	NodeRight *Node
	External  bool
}

func NewITree(X [][]float64, l float64) *ITree {
	return &ITree{
		RootNode:      NextNode(append([][]float64(nil), X...), 0, l),
		HeightLimit:   l,
		AvgPathLength: avgPathLength(float64(len(X))),
	}
}

// NextNode creates a new node in the tree
func NextNode(X [][]float64, e float64, l float64) *Node {
	// return an external node, if only one sample remains
	// or the max tree height is reached.
	if e >= l || len(X) <= 1 {
		return &Node{
			size:     len(X),
			External: true,
		}
	}

	// select a random attribute q
	q := rand.Intn(len(X[0]))

	// choose a splitpoint p between the max and min value of the attribute
	p := selectSplitPoint(append([][]float64(nil), X...), q)

	var Xl [][]float64
	var Xr [][]float64

	// split up the samples in X at the chosen split point
	for _, v := range X {
		if v[q] < p {
			Xl = append(Xl, v)
		} else {
			Xr = append(Xr, v)
		}
	}
	//log.Print(p, " --- ",len(Xl), " | ", len(Xr))

	return &Node{
		SplitPoint:     p,
		SplitAttrIndex: q,
		NodeLeft:       NextNode(Xl, e+1, l),
		NodeRight:      NextNode(Xr, e+1, l),
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

func selectSplitPoint(X [][]float64, attrIndex int) float64 {
	max := math.SmallestNonzeroFloat64
	min := math.MaxFloat64

	for i := 0; i < len(X); i++ {
		if X[i][attrIndex] > max {
			max = X[i][attrIndex]
		}
		if X[i][attrIndex] < min {
			min = X[i][attrIndex]
		}
	}

	// calculate a float in the range between min and max
	return min + rand.Float64()*(max-min)
}

func avgPathLength(n float64) float64 {
	return 2*(math.Log(n-1)+EulersConstant) - ((2 * (n - 1)) / n)
}
