
BLAS_DIR := $(HOME)/blas

CGO_ENV := CGO_LDFLAGS="-L$(HOME)/blas -lopenblas -Wl,-rpath,$(BLAS_DIR)"

run:
	go run . $(ARGS)

blas_run:
	$(CGO_ENV) go run -tags blas . $(ARGS)

test:
	go test

blas_test:
	$(CGO_ENV) go test -tags blas

build:
	go build

blas_build:
	$(CGO_ENV) go build -tags blas
