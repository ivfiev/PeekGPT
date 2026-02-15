package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

func copyModel() *transformer {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateCopyTask([]rune("0123456789"), 5, 1000, rng)
	validationSet := generateCopyTask([]rune("0123456789"), 5, 100, rng)
	t := train(32, 21, 2,
		trainingSet, validationSet,
		1, 100000, 32, 16, 0.0002, 0.00001, seed)
	return t
}

func reverseModel() *transformer {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateReverseTask([]rune("0123456789"), 4, 2500, rng)
	validationSet := generateReverseTask([]rune("0123456789"), 4, 100, rng)
	t := train(64, 9, 2,
		trainingSet, validationSet,
		4, 5000, 32, 16, 0.001, 0.0000001, seed)
	return t
}

func sumModel() *transformer {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateSumTask([]rune("0123"), 3, 1000, rng)
	validationSet := generateSumTask([]rune("0123"), 3, 100, rng)
	t := train(32, 9, 2,
		trainingSet, validationSet,
		1, 100000, 32, 16, 0.0001, 0.00001, seed)
	return t
}

func indexModel() *transformer {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateIndexTask([]rune("0123456789"), 5, 1000, rng)
	validationSet := generateIndexTask([]rune("0123456789"), 5, 100, rng)
	t := train(32, 8+3, 2,
		trainingSet, validationSet,
		4, 25000, 32, 16, 0.0005, 0.00001, seed)
	return t
}

func main() {
	blas64.Use(netlib.Implementation{})
	t := reverseModel()
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		reader := bufio.NewReader(os.Stdin)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
}
