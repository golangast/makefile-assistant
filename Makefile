## makefile-train: Train MoE model on makefile data
makefile-train:
	go run main.go -train

## export-yaml: Export training data from parent Makefile to YAML
export-yaml:
	cat ../../Makefile | go run main.go -export-yaml -yaml ../../data/training/trainingdata/makefile.yaml

## run: Run the standalone makefile assistant chat
run:
	go run main.go
## build: builds the program
build:
	go build -o app main.go

## chat: Alias for run
chat: run

## makefile-chat: Alias for run
makefile-chat: run

## fuzzy: Run the standalone fuzzy finder
fuzzy:
	go run main.go -fuzzy

## sel: Interactive fuzzy finder target selector
sel:
	@target=$$(go run main.go -fuzzy); \
	if [ -n "$$target" ]; then \
		$(MAKE) $$target; \
	fi

## help: Show available commands
help:
	@echo "Available targets:"
	@echo "  run/chat/makefile-chat  - Start interactive makefile chat"
	@echo "  fuzzy/sel               - Interactive fuzzy finder"
	@echo "  export-yaml             - Export training data from parent Makefile to YAML"
	@echo "  makefile-train          - Export YAML from parent Makefile and train"
	@echo "  help                    - Show this help"
