//go:build blas

package main

import (
	"log"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

func init() {
	log.Print("BLAS✓")
	blas64.Use(netlib.Implementation{})
}
