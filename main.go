package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"slices"
	"strings"
)

func assert(f func(*model) (any, any), label string, args ...any) {
	m := newModel(4, 3, 4, 1, 1, []rune("abcde"))
	m.rand(rand.New(rand.NewSource(7357)))
	m.loadXs([]rune("ab"))
	m.ys = []int{1, 2, -1}
	m.forward()
	target, dtarget := f(m)
	const eps = 1e-6
	switch target := target.(type) {
	case matrix:
		_, rows, cols, _ := unmat(target)
		expected := makeMat(rows, cols)
		for r := range rows {
			for c := range cols {
				target.Set(r, c, target.At(r, c)+eps)
				if len(args) == 0 || args[0].(bool) {
					if len(args) == 2 && args[1].(bool) {
						m.loadXs([]rune("dd"))
					}
					m.forward()
				}
				plus := m.loss()
				target.Set(r, c, target.At(r, c)-2*eps)
				if len(args) == 0 || args[0].(bool) {
					if len(args) == 2 && args[1].(bool) {
						m.loadXs([]rune("dd"))
					}
					m.forward()
				}
				minus := m.loss()
				expected.Set(r, c, (plus-minus)/(2*eps))
				target.Set(r, c, target.At(r, c)+eps)
			}
		}
		fmt.Printf("%s expected\n", label)
		printMat(expected)
		fmt.Printf("%s actual\n", label)
		m.forward()
		m.backward()
		printMat(dtarget.(matrix))
	case vector:
		expected := make(vector, len(target))
		for i := range target {
			target[i] += eps
			m.forward()
			plus := m.loss()
			target[i] -= 2 * eps
			m.forward()
			minus := m.loss()
			expected[i] = (plus - minus) / (2 * eps)
		}
		fmt.Printf("%s expected\n", label)
		printVec(expected)
		fmt.Printf("%s actual\n", label)
		m.forward()
		m.backward()
		printVec(dtarget.(vector))
	}
	println()
}

func main() {
	assert(func(m *model) (any, any) { return m.linear, m.dlinear }, "linear")
	assert(func(m *model) (any, any) { return m.bias2, m.dbias2 }, "bias")

	assert(func(m *model) (any, any) { return m.tokens, m.dtokens }, "tokens", true, true)
	assert(func(m *model) (any, any) { return m.positions, m.dpositions }, "positions", true, true)

	assert(func(m *model) (any, any) { return m.blocks[0].hidden, m.blocks[0].dhidden }, "hidden")
	assert(func(m *model) (any, any) { return m.blocks[0].bias1, m.blocks[0].dbias1 }, "bias1")
	assert(func(m *model) (any, any) { return m.blocks[0].input, m.blocks[0].dinput }, "input")
	assert(func(m *model) (any, any) { return m.blocks[0].bias0, m.blocks[0].dbias0 }, "bias0")

	assert(func(m *model) (any, any) { return m.blocks[0].gamma1, m.blocks[0].dgamma1 }, "gamma1")
	assert(func(m *model) (any, any) { return m.blocks[0].beta1, m.blocks[0].dbeta1 }, "beta1")
	//
	assert(func(m *model) (any, any) { return m.blocks[0].proj, m.blocks[0].dproj }, "proj")

	assert(func(m *model) (any, any) {
		return m.blocks[0].queries[0], m.blocks[0].dqueries[0]
	}, "queries")
	assert(func(m *model) (any, any) {
		return m.blocks[0].keys[0], m.blocks[0].dkeys[0]
	}, "keys")
	assert(func(m *model) (any, any) {
		return m.blocks[0].values[0], m.blocks[0].dvalues[0]
	}, "values")

	assert(func(m *model) (any, any) { return m.blocks[0].gamma0, m.blocks[0].dgamma0 }, "gamma0")
	assert(func(m *model) (any, any) { return m.blocks[0].beta0, m.blocks[0].dbeta0 }, "beta0")

	// assert(func(m *model) (any, any) { return m.L, m.dL }, "logits", false)
	// assert(func(m *model) (any, any) { return m.linear, m.dlinear }, "linear")
	// assert(func(m *model) (any, any) { return m.bias2, m.dbias2 }, "bias")
	//
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].hidden, m.blocks[len(m.blocks)-1].dhidden }, "last block's hidden")
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].bias1, m.blocks[len(m.blocks)-1].dbias1 }, "last block's bias1")
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].input, m.blocks[len(m.blocks)-1].dinput }, "last block's input")
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].bias0, m.blocks[len(m.blocks)-1].dbias0 }, "last block's bias0")
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].gamma1, m.blocks[len(m.blocks)-1].dgamma1 }, "last block's gamma1")
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].beta1, m.blocks[len(m.blocks)-1].dbeta1 }, "last block's beta1")
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].proj, m.blocks[len(m.blocks)-1].dproj }, "last block's proj")
	//
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[len(m.blocks)-1].queries[0], m.blocks[len(m.blocks)-1].dqueries[0]
	// }, "last block's queries")
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[len(m.blocks)-1].queries[1], m.blocks[len(m.blocks)-1].dqueries[1]
	// }, "last block's queries")
	//
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[len(m.blocks)-1].keys[0], m.blocks[len(m.blocks)-1].dkeys[0]
	// }, "last block's keys")
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[len(m.blocks)-1].keys[1], m.blocks[len(m.blocks)-1].dkeys[1]
	// }, "last block's keys")
	//
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[len(m.blocks)-1].values[0], m.blocks[len(m.blocks)-1].dvalues[0]
	// }, "last block's values")
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[len(m.blocks)-1].values[1], m.blocks[len(m.blocks)-1].dvalues[1]
	// }, "last block's values")
	//
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].gamma0, m.blocks[len(m.blocks)-1].dgamma0 }, "last block's gamma0")
	// assert(func(m *model) (any, any) { return m.blocks[len(m.blocks)-1].beta0, m.blocks[len(m.blocks)-1].dbeta0 }, "last block's beta0")
	//
	// assert(func(m *model) (any, any) { return m.blocks[0].gamma0, m.blocks[0].dgamma0 }, "first block's gamma0")
	// assert(func(m *model) (any, any) { return m.blocks[0].beta0, m.blocks[0].dbeta0 }, "first block's beta0")
	//
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[0].values[0], m.blocks[0].dvalues[0]
	// }, "first block's values")
	// assert(func(m *model) (any, any) {
	// 	return m.blocks[0].values[1], m.blocks[0].dvalues[1]
	// }, "first block's values")
	//
	// assert(func(m *model) (any, any) { return m.tokens, m.dtokens }, "tokens", true, true)
	// assert(func(m *model) (any, any) { return m.positions, m.dpositions }, "positions", true, true)
	// dloss/dlogits
	// dloss/dlogits * dlogits/dlinear
	// dloss/dR1 = dloss/dlogits * dlogits/dXS * dXS/dR1

	// mode := flag.String("mode", "load", "train/load/eval/gen")
	// datapath := flag.String("data", "", "training/validation data path")
	// modelpath := flag.String("model", "", "model path")
	// prompt := flag.String("prompt", "", "prompt")
	// tsize := flag.Int("t", 0, "size of the training set")
	// vsize := flag.Int("v", 0, "size of the validation set")
	// dmodel := flag.Int("dmodel", 0, "d_model")
	// context := flag.Int("ctx", 0, "context")
	// dattn := flag.Int("dattn", 0, "dattn")
	// attn := flag.Int("attn", 1, "attn")
	// blocks := flag.Int("blocks", 1, "blocks")
	// lr := flag.Float64("lr", 0.0001, "learning rate")
	// spsa := flag.Int("spsa", 8, "SPSA samples")
	// eps := flag.Float64("eps", 0.000001, "eps")
	// iters := flag.Int("iters", 1000, "training iterations")
	// ubatches := flag.Int("ub", 32, "micro-batches")
	// uiters := flag.Int("ui", 16, "micro-iters")
	// seed := flag.Int64("seed", time.Now().UnixNano(), "seed")
	// task := flag.String("task", "", "task data type")
	// vocab := flag.String("vocab", "", "vocab")
	// n := flag.Int("n", 0, "n")
	// maxLen := flag.Int("max", 0, "max")
	// flag.Parse()
	//
	// switch *mode {
	// case "load":
	// 	if *prompt == "" {
	// 		log.Panicln("empty prompt")
	// 	}
	// 	model := load(*modelpath)
	// 	model.solve([]rune(*prompt))
	// case "train":
	// 	if *dattn == 0 {
	// 		*dattn = *dmodel
	// 	}
	// 	trainingSet, validationSet := readTrainingData(*datapath, *tsize, *vsize)
	// 	model := train(
	// 		*dmodel, *context, *dattn, *attn, *blocks,
	// 		trainingSet, validationSet,
	// 		*spsa, *iters, *ubatches, *uiters, *lr, *eps,
	// 		*seed,
	// 	)
	// 	store(model, *modelpath)
	// case "gen":
	// 	if len(*vocab) == 0 {
	// 		log.Fatal("empty vocab")
	// 	}
	// 	runes := []rune(*vocab)
	// 	rng := rand.New(rand.NewSource(*seed))
	// 	switch *task {
	// 	case "copy":
	// 		for _, data := range genCopyDataset(runes, *maxLen, *n, rng) {
	// 			fmt.Println(string(data))
	// 		}
	// 	case "reverse":
	// 		for _, data := range genReverseDataset(runes, *maxLen, *n, rng) {
	// 			fmt.Println(string(data))
	// 		}
	// 	case "index":
	// 		for _, data := range genIndexDataset(runes, *maxLen, *n, rng) {
	// 			fmt.Println(string(data))
	// 		}
	// 	case "kv":
	// 		split := strings.Split(*vocab, ",")
	// 		for _, data := range genKVdataset([]rune(split[0]), []rune(split[1]), *maxLen, *n, rng) {
	// 			fmt.Println(string(data))
	// 		}
	// 	default:
	// 		log.Fatalf("unknown task %s", *task)
	// 	}
	// case "eval":
	// 	model := load(*modelpath)
	// 	validationSet, _ := readTrainingData(*datapath, *vsize, 0)
	// 	tr := newTraining(model)
	// 	tr.validation = validationSet
	// 	loss := tr.validate(model)
	// 	fmt.Printf("Loss: %.12f\n", loss)
	// 	fmt.Printf("Prob: %.12f\n", math.Exp(-loss))
	// default:
	// 	log.Fatalf("unknown mode %s", *mode)
	// }
}

func readTrainingData(path string, t, v int) ([][]rune, [][]rune) {
	data := make([][]rune, 0)
	bytes, err := os.ReadFile(path)
	if err != nil {
		log.Panic(err)
	}
	words := strings.SplitSeq(string(bytes), "\n")
	for word := range words {
		data = append(data, []rune(strings.TrimSpace(word)))
	}
	return data[:t], data[t : t+v]
}

type stored struct {
	DModel  int
	Context int
	DAttn   int
	Attn    int
	Blocks  int
	Vocab   []rune
	Params  vector
}

func store(m *model, path string) {
	stored := &stored{}
	stored.DModel = m.dModel
	stored.Context = m.context
	stored.DAttn = m.blocks[0].dAttn
	stored.Attn = m.blocks[0].attn
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
	model := newModel(stored.DModel, stored.Context, stored.DAttn, stored.Attn, stored.Blocks, stored.Vocab)
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
		q := 1 // + rng.Int()%kv
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
		dataset = append(dataset, []rune(fmt.Sprintf("%s%s|%s=%s", string(query), string(dict), strings.Repeat("?", len(query)), string(answer))))
	}
	return dataset
}
