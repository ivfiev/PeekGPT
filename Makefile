
CGO_ENV := LD_LIBRARY_PATH=$(HOME)/blas:$$LD_LIBRARY_PATH \
	CGO_LDFLAGS="-L$(HOME)/blas -lopenblas" \

netlib:
	$(CGO_ENV) go install gonum.org/v1/netlib/blas/netlib

run:
	$(CGO_ENV) go run .

test:
	$(CGO_ENV) go test

build:
	$(CGO_ENV) go build
