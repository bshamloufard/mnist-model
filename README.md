# Digit Recognition

This project is a MNIST recognition model

## How To Use

You need to use the test_prediction function.

```bash
# Function and it's parameters
test_prediction(index, params, X.T, Y)

# Example functions, index can be any number from 2-28001
# params is already defined, X and Y are also already defined
test_prediction(231, params, X, Y)
test_prediction(32, params, X, Y)
test_prediction(10801, params, X, Y)
```