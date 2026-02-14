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

func copyModel() {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateCopyTask([]rune("0123456789"), 5, 1000, rng)
	validationSet := generateCopyTask([]rune("0123456789"), 5, 100, rng)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	t := train(32, 21, 2,
		trainingSet, validationSet,
		100000, 32, 16, 0.0002, 0.00001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
}

func reverseModel() {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateReverseTask([]rune("0123456789"), 4, 1000, rng)
	validationSet := generateReverseTask([]rune("0123456789"), 4, 100, rng)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	t := train(32, 9, 2,
		trainingSet, validationSet,
		100000, 32, 16, 0.0001, 0.00001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
}

func indexModel() {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateIndexTask([]rune("0123456789"), 8, 10000, rng)
	validationSet := generateIndexTask([]rune("0123456789"), 8, 100, rng)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	t := train(32, 8+3, 3,
		trainingSet, validationSet,
		100000, 32, 16, 0.0001, 0.00001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
}

func sumModel() {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := generateSumTask([]rune("0123"), 3, 1000, rng)
	validationSet := generateSumTask([]rune("0123"), 3, 100, rng)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	t := train(32, 9, 2,
		trainingSet, validationSet,
		100000, 32, 16, 0.0001, 0.00001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
}

func main() {
	indexModel()
}
