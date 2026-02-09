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

func generateData(vocab []rune, n int) {
	for range n {
		count := 1 + rand.Int()%2
		example := make([]rune, 0, count)
		for range count {
			example = append(example, vocab[rand.Int()%len(vocab)])
		}
		mask := strings.Repeat("?", count)
		fmt.Printf("[]rune(\"%s|%s=%s\"),\n", string(example), mask, string(example))
	}
}

func main() {
	generateData([]rune("abcd"), 50)
	blas64.Use(netlib.Implementation{})
	reader := bufio.NewReader(os.Stdin)
	seed := time.Now().UnixNano()
	t := train(32, 6, 10,
		[][]rune{
			[]rune("bb|??=bb"),
			[]rune("cc|??=cc"),
			[]rune("ab|??=ab"),
			[]rune("dc|??=dc"),
			[]rune("d|?=d"),
			[]rune("a|?=a"),
			[]rune("ac|??=ac"),
			[]rune("ba|??=ba"),
			[]rune("bd|??=bd"),
			[]rune("cd|??=cd"),
			[]rune("b|?=b"),
			[]rune("dd|??=dd"),
			[]rune("da|??=da"),
			[]rune("dd|??=dd"),
			[]rune("ba|??=ba"),
			[]rune("c|?=c"),
			[]rune("c|?=c"),
			[]rune("ab|??=ab"),
			[]rune("b|?=b"),
			[]rune("ad|??=ad"),
			[]rune("b|?=b"),
			[]rune("dd|??=dd"),
			[]rune("ad|??=ad"),
			[]rune("dc|??=dc"),
			[]rune("db|??=db"),
			[]rune("c|?=c"),
			[]rune("bc|??=bc"),
			[]rune("bc|??=bc"),
			[]rune("c|?=c"),
			[]rune("bb|??=bb"),
			[]rune("b|?=b"),
			[]rune("bb|??=bb"),
			[]rune("a|?=a"),
			[]rune("ca|??=ca"),
			[]rune("dc|??=dc"),
			[]rune("dc|??=dc"),
			[]rune("ad|??=ad"),
			[]rune("aa|??=aa"),
			[]rune("dc|??=dc"),
			[]rune("aa|??=aa"),
			[]rune("ca|??=ca"),
		},
		50000, 4, 16, 0.00025, 0.0001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		// t.peek([]rune(strings.TrimRight(input, "\n\r")))
		t.solve([]rune(strings.TrimRight(input, "\n\r")))
	}
	// task mode, -1 in ys, try to make compatible with seqs
	// generate data examples better
	// fix graphics/peek
	// validation + training set, use that shuffle func
	// multi-blocks
}
