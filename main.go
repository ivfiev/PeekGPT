package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"slices"
	"strings"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

func main() {
	blas64.Use(netlib.Implementation{})

	mode := flag.String("mode", "load", "train/load")
	datapath := flag.String("data", "", "training/validation data path")
	modelpath := flag.String("model", "", "model path")
	prompt := flag.String("prompt", "", "prompt")
	tsize := flag.Int("t", 0, "size of the training set")
	vsize := flag.Int("v", 0, "size of the validation set")
	dmodel := flag.Int("dmodel", 0, "d_model")
	context := flag.Int("ctx", 0, "context")
	blocks := flag.Int("blocks", 1, "blocks")
	lr := flag.Float64("lr", 0.0001, "learning rate")
	spsa := flag.Int("spsa", 8, "SPSA samples")
	eps := flag.Float64("eps", 0.000001, "eps")
	iters := flag.Int("iters", 1000, "training iterations")
	ubatches := flag.Int("ub", 32, "micro-batches")
	uiters := flag.Int("ui", 16, "micro-iters")
	seed := flag.Int64("seed", time.Now().UnixNano(), "seed")
	flag.Parse()

	switch *mode {
	case "load":
		if *prompt == "" {
			log.Panicln("empty prompt")
		}
		model := load(*modelpath)
		model.solve([]rune(*prompt))
	case "train":
		trainingSet, validationSet := readTrainingData(*datapath, *tsize, *vsize)
		model := train(*dmodel, *context, *blocks, trainingSet, validationSet, *spsa, *iters, *ubatches, *uiters, *lr, *eps, *seed)
		store(model, *modelpath)
	}

	// model := reverseModel()
	// for {
	// 	fmt.Printf("Enter context, up to %d chars: ", model.context)
	// 	reader := bufio.NewReader(os.Stdin)
	// 	input, err := reader.ReadString('\n')
	// 	if err != nil {
	// 		log.Fatal(err)
	// 	}
	// 	model.solve([]rune(strings.TrimRight(input, "\n\r")))
	// }
}

func readTrainingData(path string, t, v int) ([][]rune, [][]rune) {
	data := make([][]rune, 0)
	bytes, err := os.ReadFile(path)
	if err != nil {
		log.Panic(err)
	}
	words := strings.Split(string(bytes), "\n")
	for _, word := range words {
		data = append(data, []rune(strings.TrimSpace(word)))
	}
	return data[:t], data[t : t+v]
}

type stored struct {
	DModel  int
	Context int
	Blocks  int
	Vocab   []rune
	Params  vector
}

func store(m *model, path string) {
	stored := &stored{}
	stored.DModel = m.dModel
	stored.Context = m.context
	stored.Blocks = len(m.blocks)
	stored.Vocab = m.vocab
	stored.Params = make(vector, m.size())
	m.dump(stored.Params)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o666)
	if err != nil {
		log.Panic(err)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.Encode(stored)
}

func load(path string) *model {
	stored := &stored{}
	file, err := os.OpenFile(path, os.O_RDONLY, 0o777)
	if err != nil {
		log.Panic(err)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.Decode(stored)
	model := newModel(stored.DModel, stored.Context, stored.Blocks, stored.Vocab)
	model.apply(stored.Params)
	return model
}

func genCopyDataset(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		data := make([]rune, k)
		for i := range k {
			data[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(data)
		qs := strings.Repeat("?", len(str))
		dataset = append(dataset, []rune(fmt.Sprintf("%s|%s=%s", str, qs, str)))
	}
	return dataset
}

func genReverseDataset(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		data := make([]rune, k)
		for i := range k {
			data[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(data)
		slices.Reverse(data)
		rev := string(data)
		qs := strings.Repeat("?", len(str))
		dataset = append(dataset, []rune(fmt.Sprintf("%s|%s=%s", str, qs, rev)))
	}
	return dataset
}

func genIndexDataset(vocab []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	for range n {
		k := 1 + rng.Int()%maxLen
		data := make([]rune, k)
		for i := range k {
			data[i] = vocab[rng.Int()%len(vocab)]
		}
		str := string(data)
		ix := rng.Int() % k
		ch := data[ix]
		dataset = append(dataset, []rune(fmt.Sprintf("%d%s|?=%c", ix, str, ch)))
	}
	return dataset
}
