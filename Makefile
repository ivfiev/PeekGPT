
BLAS_DIR := $(HOME)/blas

CGO_ENV := LD_LIBRARY_PATH=$(BLAS_DIR):$$LD_LIBRARY_PATH \
	CGO_LDFLAGS="-L$(HOME)/blas -lopenblas -Wl,-rpath,$(BLAS_DIR)" \

netlib:
	$(CGO_ENV) go install gonum.org/v1/netlib/blas/netlib

run:
	$(CGO_ENV) go run . $(ARGS)

test:
	$(CGO_ENV) go test

build:
	$(CGO_ENV) go build
