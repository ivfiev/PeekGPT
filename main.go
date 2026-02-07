package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	seed := time.Now().UnixNano()
	t := train(27, 16,
		[]rune("the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog the quick brown fox jumps over the lazy dog "),
		50000, 16, 16, 0.0004, 0.0001, seed)
	for {
		fmt.Printf("Enter context, up to %d chars: ", t.context)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		t.peek([]rune(strings.TrimRight(input, "\n\r")))
	}
	// learnable embeds
}

// package main
//
// import (
// 	"fmt"
// 	"math/rand"
//
// 	"gonum.org/v1/gonum/blas/blas64"
// 	"gonum.org/v1/gonum/mat"
// 	"gonum.org/v1/netlib/blas/netlib"
// )
//
// // func main() {
// // 	blas64.Use(netlib.Implementation{})
// // 	impl := blas64.Implementation()
// // 	fmt.Printf("BLAS implementation: %T\n", impl)
// // 	size := 64
// // 	data1 := make(vector, size*size)
// // 	data2 := make(vector, size*size)
// // 	for i := range data2 {
// // 		data1[i] = rand.Float64()
// // 		data2[i] = rand.Float64()
// // 	}
// // 	// a := mat.NewDense(size, size, make(vector, size*size))
// // 	b := mat.NewDense(size, size, data1)
// // 	// c := mat.NewDense(size, size, data2)
// //
// // 	for range 1000000 {
// // 		b.Mul(b, b)
// // 	}
// //
// // 	// my := makeMat(size, size)
// // 	// for range 1000000 {
// // 	// 	mulMatT(my, my, my)
// // 	// }
// // }
//
// // export CGO_LDFLAGS="-L$HOME/dev/tiny-transformers/blas -lopenblas"
// // export CGO_CFLAGS="-I$HOME/dev/tiny-transformers/blas/include"
// // export LD_LIBRARY_PATH="$HOME/dev/tiny-transformers/blas:$LD_LIBRARY_PATH"
