package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

type quad struct {
	v vector
}

func (q *quad) Eval(u vector) scalar {
	if len(q.v) != len(u) {
		log.Fatal("quad.Eval: incompatible vectors")
	}
	sum := 0.0
	for i := range len(u) {
		sum += math.Pow(float64(q.v[i]-u[i]), 2)
	}
	return scalar(sum)
}

func (q *quad) Clone() Objective {
	v := make([]scalar, len(q.v))
	copy(v, q.v)
	return &quad{v: v}
}

func (q *quad) Size() int {
	return len(q.v)
}

func run() {
	const size = 1000
	v := make(vector, size)
	u := make(vector, size)
	for i := range v {
		v[i] = scalar(10 * (rand.Float32() - 0.5))
		u[i] = scalar(20 * (rand.Float32() - 0.5))
	}
	q := quad{v: v}
	spsa(&q, u, 10000, 2, 8, 0.001, 0.001)
	fmt.Printf("%v\n%f\n", u, q.Eval(u))
}
