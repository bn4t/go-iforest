# go-iforest
[![godocs.io](https://godocs.io/github.com/bn4t/go-iforest?status.svg)](https://godocs.io/github.com/bn4t/go-iforest)

Golang implementation of [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).

## Example Usage
```go
package main

import (
	"fmt"
	"log"
	"math/rand"

	go_iforest "github.com/bn4t/go-iforest"
)

func main() {
	// seed the random generator before generating the isolation forest
	rand.Seed(42)

	// generate 1000 samples with two normally distributed features
	inputData := make([][]float64, 1000)
	for i := range inputData {
		inputData[i] = []float64{rand.NormFloat64(), rand.NormFloat64()}
	}

	// generate an isolation forest with 100 trees and a sub-sampling size of 256
	f, err := go_iforest.NewIForest(inputData, 100, 256)
	if err != nil {
		log.Fatal(err)
	}

	// scores close to 1 indicate anomalies, scores around 0.5 are normal
	fmt.Println(f.CalculateAnomalyScore(inputData[0]))    // around 0.5
	fmt.Println(f.CalculateAnomalyScore([]float64{8, 8})) // clearly anomalous
}
```

The sub-sampling size must be at least 2 and at most the amount of provided samples.

## License

MIT
