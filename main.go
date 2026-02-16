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

func reverseModel() *model {
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	trainingSet := genReverseDataset([]rune("0123456789"), 10, 25000, rng)
	validationSet := genReverseDataset([]rune("0123456789"), 10, 100, rng)
	return train(64, 21, 2,
		trainingSet, validationSet,
		8, 15000, 32, 16, 0.001, 0.0000001, seed)
}

// TODO generate datasets using data funcs & delete data.go

func main() {
	blas64.Use(netlib.Implementation{})
	model := reverseModel()
	for {
		fmt.Printf("Enter context, up to %d chars: ", model.context)
		reader := bufio.NewReader(os.Stdin)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		model.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
}

type stored struct {
	dModel  int
	context int
	blocks  int
	vocab   []rune
	params  vector
}

func store(m *model) {
}

func load(file string) *model {
	return nil
}
