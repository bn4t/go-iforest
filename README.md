# go-iforest
[![godocs.io](https://godocs.io/github.com/codegaudi/go-iforest?status.svg)](https://godocs.io/github.com/codegaudi/go-iforest)

Golang implementation of [Isolation Forests](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).

## Example Usage
```go
package main

import (
	go_iforest "github.com/codegaudi/go-iforest"
	"log"
	"math/rand"
	"time"
)


func main() {

	// load input data into two-dimensional slice
	var inputData [][]float64
	
	// seed the random generator before generating an isolation forest
	rand.Seed(time.Now().UnixNano())

	// generate an isolation forest with 10'000 trees and a sub-sampling size of 256
	f, err := go_iforest.NewIForest(inputData, 10000, 256)
	if err != nil {
		log.Fatal(err)
	}

	// calculate an anomaly score for a sample
	score := f.CalculateAnomalyScore(inputData[0])
}
```

## License

MIT