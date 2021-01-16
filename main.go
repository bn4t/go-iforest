package main

import (
	go_iforest "go-iforest/go-iforest"
	"log"
	"math"
	"math/rand"
	"time"
)

// TODO: rewrite to not modify [][]X

func main() {
	rand.Seed(time.Now().UnixNano())

	var inputData [][]float64
	var sum float64

	for i := 0; i < 5000; i++ {
		a := math.Pow(float64(100+rand.Intn(100))+rand.Float64(), 4)
		log.Print(a)
		sum+=a
		inputData = append(inputData, []float64{a})
	}

	log.Print("avg: ", sum/5000)


	n, err := go_iforest.NewIForest(inputData, 1000, 256)
	if err != nil {
		log.Fatal(err)
	}

	log.Print("================")
	a := 0.1
	b := 1000000000000000000000000000.0
	log.Print(a, " | ",n.CalculateAnomalyScore([]float64{a}))
	log.Print(b," | ", n.CalculateAnomalyScore([]float64{b}))


	return
}
