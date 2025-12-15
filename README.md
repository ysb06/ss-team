# System Security Team Project

This project is a security research initiative that reproduces the **PROMPTPEEK** side-channel attack. It analyzes security vulnerabilities that exploit KV cache sharing mechanisms and LPM (Longest Prefix Match) scheduling policies in multi-tenant LLM serving frameworks.

**⚠️ Important**: This code should be used for educational and research purposes only. Unauthorized attacks on production environments are illegal and ethically problematic.

## Project Overview

This project consists of three main components:

- **LPM Server**: A mock LLM server implementing LPM (Longest Prefix Match) scheduling
- **Serverless System**: A Docker-based serverless function execution environment (implemented in Go)
- **PromptPeek**: A client application that performs the actual attack

## Prerequisites

### Requirements

- **Python**: 3.11.x
- **Go**: 1.25.3 or higher
- **Docker**: Required for container execution
- **Operating System**: macOS, Linux, or Windows

### Python Virtual Environment Setup

Before running the project for the first time, set up a Python virtual environment and install the required packages:

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## How to Run

Each component of the project runs in a separate terminal. Open each terminal in the order below and execute the commands.

### 1. Running the LPM Server

The LPM server acts as a mock server that handles LLM requests. This server implements the LPM scheduling algorithm to simulate cache hit patterns.

**Open a new terminal** and run the following commands:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start LPM server
PYTHONPATH=./src python -m test_peek
```

Once the server starts successfully, it will wait for requests on a local port.

### 2. Running the Serverless System

The serverless system provides a Docker-based function execution environment. This component dynamically manages function instances and supports various startup policies (cold/hot start).

**Open a new terminal** and run the following commands:

```bash
# Start serverless runtime
go run . --port 8080 --config ./example_config.json
```

Configuration file options:
- `example_config.json`: Default configuration
- `example_config_cold.json`: Cold start policy
- `example_config_hot.json`: Hot start policy

### 3. Running the PromptPeek Attack Client

PromptPeek is the client that performs the actual attack. It sends multiple probe requests and analyzes response times to infer the victim's prompts.

**Open a new terminal** and run the following commands:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start PromptPeek attack
PYTHONPATH=./src python -m promptpeek
```

Once the attack starts, it will load test prompts from the `data/prompts.csv` file and attempt to infer them token by token.

## Project Structure

```
ss-team/
├── src/
│   ├── promptpeek/          # Attack client implementation
│   │   ├── attack.py         # Main attack logic
│   │   ├── candidate/        # Candidate token generation
│   │   └── dataset.py        # Dataset loader
│   └── test_peek/            # LPM server implementation
│       └── lpm.py            # LPM scheduling logic
├── internal/                 # Go serverless system
│   ├── policy/               # Startup policy implementation
│   ├── slrun/                # Runtime logic
│   └── types/                # Type definitions
├── functions/                # Example serverless functions
├── data/                     # Test data
│   └── prompts.csv           # Test prompt dataset
└── IMPLEMENTATION_GUIDELINES.md  # Detailed implementation guide
```

## Additional Resources

- Detailed implementation guidelines: [IMPLEMENTATION_GUIDELINES.md](IMPLEMENTATION_GUIDELINES.md)
- Legacy documentation: [README_legacy.md](README_legacy.md)

## Troubleshooting

### Python Module Not Found
Make sure you've added `PYTHONPATH=./src` before your command.

### Docker Permission Error
Running the serverless system requires access to the Docker daemon. Verify that your user is part of the docker group.

### Port Conflict Issue
If the default port is already in use, you can specify a different port using the `--port` flag.