package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"slices"
	"strings"
	"time"
)

func main() {
	// f, err := os.Create("cpu.prof")
	// if err != nil {
	// 	panic(err)
	// }
	// pprof.StartCPUProfile(f)
	// defer pprof.StopCPUProfile()

	mode := flag.String("mode", "load", "train/solve/eval (loss)/gen (data)/prompt")
	datapath := flag.String("data", "", "training/validation data path")
	modelpath := flag.String("model", "", "model path")
	checkpoint := flag.Bool("checkpoint", false, "continue training existing model")
	prompt := flag.String("prompt", "", "prompt")
	tsize := flag.Int("t", 0, "size of the training set")
	vsize := flag.Int("v", 0, "size of the validation set")
	dmodel := flag.Int("dmodel", 0, "d_model")
	context := flag.Int("ctx", 0, "context")
	dattn := flag.Int("dattn", 0, "dattn")
	attn := flag.Int("attn", 1, "attn")
	blocks := flag.Int("blocks", 1, "blocks")
	lr := flag.Float64("lr", 0.001, "learning rate")
	iters := flag.Int("iters", 1000, "training iterations")
	ubatches := flag.Int("ub", 64, "micro-batches")
	seed := flag.Int64("seed", time.Now().UnixNano(), "seed")
	task := flag.String("task", "", "task data type")
	vocab := flag.String("vocab", "", "vocab")
	n := flag.Int("n", 0, "n")
	maxLen := flag.Int("max", 0, "max")
	textmode := flag.Bool("text", false, "text generation mode")
	par := flag.Int("par", 8, "parallel training")
	mlp := flag.Int("mlp", 2, "MLP width")
	steps := flag.Int("steps", 100, "validation frequency every x iters")
	flag.Parse()

	switch *mode {
	case "solve":
		if *prompt == "" {
			log.Panicln("empty prompt")
		}
		model := load(*modelpath)
		ctx := []rune(*prompt)
		ys := []int{}
		if *textmode {
			ys = append(ys, len(ctx)-1)
		} else {
			for i := range ctx {
				if ctx[i] == '?' {
					ys = append(ys, i)
				}
			}
		}
		model.solve(ctx, ys)

	case "prompt":
		if *prompt == "" {
			log.Panicln("empty prompt")
		}
		model := load(*modelpath)
		model.generate([]rune(*prompt), *n)

	case "train":
		if *dattn == 0 {
			*dattn = *dmodel / *attn
		}
		var model *model
		if *checkpoint {
			model = load(*modelpath)
			log.Printf("Loaded checkpoint [%s]\n", *modelpath)
		}
		trainingSet, validationSet := readTrainingData(*datapath, *tsize, *vsize, *textmode)
		model = train(
			*dmodel, *context, *dattn, *attn, *mlp, *blocks,
			trainingSet, validationSet, *steps,
			*iters, *ubatches, *par, *lr,
			*seed,
			model,
		)
		store(model, *modelpath)

	case "gen":
		if len(*vocab) == 0 {
			log.Fatal("empty vocab")
		}
		runes := []rune(*vocab)
		rng := rand.New(rand.NewSource(*seed))
		switch *task {
		case "copy":
			for _, data := range genCopyDataset(runes, *maxLen, *n, rng) {
				fmt.Println(string(data))
			}
		case "reverse":
			for _, data := range genReverseDataset(runes, *maxLen, *n, rng) {
				fmt.Println(string(data))
			}
		case "index":
			for _, data := range genIndexDataset(runes, *maxLen, *n, rng) {
				fmt.Println(string(data))
			}
		case "kv":
			split := strings.Split(*vocab, ",")
			for _, data := range genKVdataset([]rune(split[0]), []rune(split[1]), *maxLen, *n, rng) {
				fmt.Println(string(data))
			}
		default:
			log.Fatalf("unknown task %s", *task)
		}

	case "eval":
		model := load(*modelpath)
		validationSet, _ := readTrainingData(*datapath, *vsize, 0, *textmode)
		tr := newTrainer(model, *par)
		tr.validation = validationSet
		loss := tr.validate(model)
		fmt.Printf("Loss: %.12f\n", loss)
		fmt.Printf("Prob: %.12f\n", math.Exp(-loss))

	default:
		log.Fatalf("unknown mode %s", *mode)
	}
}

func readTrainingData(path string, t, v int, textmode bool) ([][]rune, [][]rune) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		log.Panic(err)
	}
	if textmode {
		runes := []rune(string(bytes))
		trainingSet, validationSet := [][]rune{runes[v:]}, [][]rune{runes[:v]}
		return trainingSet, validationSet
	} else {
		data := make([][]rune, 0)
		words := strings.SplitSeq(string(bytes), "\n")
		for word := range words {
			data = append(data, []rune(strings.TrimSpace(word)))
		}
		return data[:t], data[t : t+v]
	}
}

type stored struct {
	DModel  int
	Context int
	DAttn   int
	Attn    int
	Mlp     int
	Blocks  int
	Vocab   []rune
	Params  vector
}

func store(m *model, path string) {
	stored := &stored{}
	stored.DModel = m.dModel
	stored.Context = m.context
	stored.DAttn = m.dAttn
	stored.Attn = m.attn
	stored.Mlp = m.mlp
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
	log.Printf("Stored %s\n", path)
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
	model := newModel(stored.DModel, stored.Context, stored.DAttn, stored.Attn, stored.Mlp, stored.Blocks, stored.Vocab)
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

func genKVdataset(vocabK, vocabV []rune, maxLen, n int, rng *rand.Rand) [][]rune {
	dataset := make([][]rune, 0, n)
	indexes := make([]int, 1+maxLen)
	randIxs := func(n, m int) []int {
		for i := range m {
			indexes[i] = i
		}
		rng.Shuffle(m, func(i, j int) {
			indexes[i], indexes[j] = indexes[j], indexes[i]
		})
		return indexes[:n]
	}
	for range n {
		kv := 1 + rng.Int()%maxLen
		q := 1 + rng.Int()%kv
		ixs := randIxs(kv, len(vocabK))
		dict := make([]rune, 2*kv)
		query := make([]rune, q)
		answer := make([]rune, q)
		for i := range kv {
			dict[2*i] = vocabK[ixs[i]]
			dict[2*i+1] = vocabV[rng.Int()%len(vocabV)]
		}
		ixs = randIxs(q, kv)
		for i := range query {
			query[i] = dict[2*ixs[i]]
			answer[i] = dict[1+slices.Index(dict, query[i])]
		}
		dataset = append(dataset, []rune(fmt.Sprintf("%s,%s|%s=%s", string(dict), string(query), strings.Repeat("?", len(query)), string(answer))))
	}
	return dataset
}
