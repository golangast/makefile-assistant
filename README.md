# Makefile Assistant (Gollemer)

**Makefile Assistant** parses your project’s `Makefile`, trains an AI model on its targets, and lets you execute commands via an interactive terminal UI or natural language chat.

---

## Prerequisites

* **Go 1.26+** (requires native SIMD acceleration)

---

## Quick Start

### 1. Installation

Clone the repository and set up the build environment:

```bash
git clone [https://github.com/golangast/makefile-assistant](https://github.com/golangast/makefile-assistant)
cd gollemer
export GOEXPERIMENT=simd
go mod tidy
```

```bash

2. Export Makefile Data
Reads your Makefile and exports the targets into YAML format for processing:
make export-yaml

```
```bash
3. Train the Model
Trains the internal model on your parsed Makefile commands:
make makefile-train

```
```bash
4. Interactive Terminal UI
Opens a Terminal UI to visually browse and run commands:
make sel

```
```bash
5. Chat Interface
Starts a conversational interface to trigger commands using natural language:
make makefile-chat
Example: "Hey, run the makefile training"
```
```bash

How Chat Matching Works
When using make makefile-chat, the assistant analyzes your natural language input and evaluates candidate commands:
```
```bash

Direct Match: If one command clearly matches your request, it executes directly.

Tied Matches: If multiple commands receive identical top confidence scores (for example, both makefile-train and makefile-chat hit 100% confidence), the system automatically opens the Terminal UI (make sel) filtered to display only those matching choices.
```

