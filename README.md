# makefile-assistant

## ⚡ Quick Start

### 1. Installation
Gollemer requires **Go 1.26+** to leverage native SIMD acceleration.

```bash
git clone https://github.com/golangast/makefile-assistant
cd gollemer
export GOEXPERIMENT=simd
go mod tidy
```

### 2. Training the Model
We provide a simplified `Makefile` to handle the curriculum training process.

```bash
make export-yaml ##to read your make file
make makefile-train ##to train the model
make sel ## to see all commands and run them in terminal UI
make makefile-chat ## to talk to the program to choose your makefile command (example: hey run the makefile training)
```
### Things to know
The sel is the terminal UI but if the command you give to makefile-chat gets both 100%'s then it will run the UI and only show those two commands
example: You: run the makefile training
This may bring up [makefile-train] [makefile-chat] because both got 100% guesses so it shows both to choose from.
