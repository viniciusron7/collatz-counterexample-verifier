# Collatz counterexample verifier and candidate generator — requires GMP (macOS: brew install gmp)
CC       ?= cc
GMP      ?= $(shell brew --prefix gmp 2>/dev/null || echo /usr/local)
CFLAGS   ?= -O3 -mcpu=native -std=gnu11 -Wall -Wextra
CPPFLAGS += -I$(GMP)/include
# Link GMP statically when possible (binary independent of the local installation).
GMPLIB   := $(if $(wildcard $(GMP)/lib/libgmp.a),$(GMP)/lib/libgmp.a,-L$(GMP)/lib -lgmp)

all: collatz candidates

collatz: collatz.c tested.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(GMPLIB) -lm -pthread

candidates: candidates.c tested.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(GMPLIB) -lm -pthread

test: collatz candidates
	python3 tests/test_collatz.py
	python3 tests/test_candidates.py

test-candidates: candidates
	python3 tests/test_candidates.py

# Build Tested/data.sqlite from Tested/data.csv (see tools/data_to_sqlite.py).
sqlite:
	python3 tools/data_to_sqlite.py

bench: collatz
	@if [ -f bench/bench.py ]; then python3 bench/bench.py; \
	else echo "make bench: bench/bench.py not found."; exit 1; fi

clean:
	rm -f collatz candidates

.PHONY: all test test-candidates sqlite bench clean
