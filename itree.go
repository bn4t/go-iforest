package go_iforest

import (
	"math"
	"math/rand"
)

// EulersConstant is Euler's constant, used by the average path length formula.
const EulersConstant = 0.5772156649

// ITree is a single isolation tree.
type ITree struct {
	// The root node of the tree
	// From this node all child nodes can be accessed by
	// following the Node{nodeLeft,nodeRight} variables
	rootNode *Node

	// amount of features the tree was trained on
	numFeatures int

	// The height limit of the tree, calculated using
	// ceiling(log2(v)) where v is the sub-sampling size
	heightLimit float64

	// The average path length of this tree, calculated
	// using avgPathLength()
	avgPathLength float64
}

// Node is a node in an isolation tree. External nodes are leaves.
type Node struct {
	// size of the remaining samples if this is an external node.
	// This is relevant if the max tree size is reached
	size int

	// the split point of this node
	splitPoint float64

	// index of the attribute used for the split point
	splitAttrIndex int

	nodeLeft  *Node
	nodeRight *Node
	external  bool
}

// NewITree creates an isolation tree from the provided samples.
// All samples must have the same amount of features.
func NewITree(X [][]float64) *ITree {
	n := len(X)
	numFeatures := 0
	if n > 0 {
		numFeatures = len(X[0])
	}

	l := 0.0
	if n > 1 {
		l = math.Ceil(math.Log2(float64(n)))
	}

	var root *Node
	if n == 0 || numFeatures == 0 {
		root = &Node{size: n, external: true}
	} else {
		var indices []int
		for k := range X {
			indices = append(indices, k)
		}
		root = nextNode(X, indices, 0, l)
	}

	return &ITree{
		rootNode:      root,
		numFeatures:   numFeatures,
		heightLimit:   l,
		avgPathLength: avgPathLength(float64(n)),
	}
}

// PathLength derives the path length of a sample x in this tree.
// It returns NaN if x does not have the same amount of features as the
// samples the tree was trained on.
func (t *ITree) PathLength(x []float64) float64 {
	if t == nil || t.rootNode == nil || len(x) != t.numFeatures {
		return math.NaN()
	}

	return PathLength(x, t.rootNode, 0)
}

// nextNode creates a new node in the tree
func nextNode(X [][]float64, indices []int, e float64, l float64) *Node {
	// return an external node, if only one sample remains
	// or the max tree height is reached.
	if e >= l || len(indices) <= 1 {
		return &Node{
			size:     len(indices),
			external: true,
		}
	}

	// select a random attribute q, skipping attributes that do not vary
	// across the remaining samples.
	q, min, max, ok := selectSplitAttribute(X, indices)
	if !ok {
		// none of the attributes can split the remaining samples
		return &Node{
			size:     len(indices),
			external: true,
		}
	}

	// choose a split point p between the max and min value of the attribute
	var p float64
	if span := max - min; math.IsInf(span, 0) {
		// max-min overflows for values near the float64 limits; split in the
		// middle instead of producing an infinite split point.
		p = min/2 + max/2
	} else {
		p = min + rand.Float64()*span
	}

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

	// A split point can round to one of the attribute's bounds, in which case
	// it does not separate the samples at all. Treat the node as a leaf instead
	// of building a chain of useless nodes.
	if len(IndicesL) == 0 || len(IndicesR) == 0 {
		return &Node{
			size:     len(indices),
			external: true,
		}
	}

	return &Node{
		splitPoint:     p,
		splitAttrIndex: q,
		nodeLeft:       nextNode(X, IndicesL, e+1, l),
		nodeRight:      nextNode(X, IndicesR, e+1, l),
		external:       false,
	}
}

// PathLength derives the path length for a sample x starting at node T.
// e is the current path length (0 on the first call to this function).
func PathLength(x []float64, T *Node, e int) float64 {
	if T == nil {
		return float64(e)
	}

	if T.external {
		if T.size <= 1 {
			return float64(e)
		} else {
			// return the current path length plus an adjustment of c(size), where size is the
			// size of the remaining samples when this external node was created (either because
			// the tree size limit was reached, or because only a single sample was remaining ->
			// size = 1). c is the avgPathLength function.
			return float64(e) + avgPathLength(float64(T.size))
		}
	}

	if x[T.splitAttrIndex] < T.splitPoint {
		return PathLength(x, T.nodeLeft, e+1)
	} else {
		return PathLength(x, T.nodeRight, e+1)
	}
}

// selectSplitAttribute picks a random attribute that varies across the
// samples at the provided indices and returns its minimum and maximum value.
// ok is false if every attribute is constant, meaning no split is possible.
func selectSplitAttribute(X [][]float64, indices []int) (attrIndex int, min, max float64, ok bool) {
	numAttributes := len(X[0])

	// Probe random attributes first, which is cheap when most attributes vary.
	for attempt := 0; attempt < numAttributes; attempt++ {
		attrIndex = rand.Intn(numAttributes)
		min, max = attributeRange(X, indices, attrIndex)
		if min < max {
			return attrIndex, min, max, true
		}
	}

	// Probing can miss a varying attribute when several attributes are
	// constant. Check all of them and choose one of the varying attributes
	// uniformly at random.
	varying := 0
	for i := 0; i < numAttributes; i++ {
		candidateMin, candidateMax := attributeRange(X, indices, i)
		if candidateMin < candidateMax {
			varying++
			if rand.Intn(varying) == 0 {
				attrIndex, min, max = i, candidateMin, candidateMax
			}
		}
	}

	if varying > 0 {
		return attrIndex, min, max, true
	}

	return 0, 0, 0, false
}

// attributeRange returns the min and max value of attrIndex for the samples
// at the provided indices.
func attributeRange(X [][]float64, indices []int, attrIndex int) (min, max float64) {
	min = X[indices[0]][attrIndex]
	max = min

	for _, v := range indices[1:] {
		value := X[v][attrIndex]
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}

	return min, max
}

// avgPathLength returns c(n), the average path length of an unsuccessful
// search in a binary search tree built from n samples.
func avgPathLength(n float64) float64 {
	if n <= 1 {
		return 0
	}

	if n == 2 {
		return 1
	}

	return 2*(math.Log(n-1)+EulersConstant) - ((2 * (n - 1)) / n)
}
