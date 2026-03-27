from keras_model import model
import numpy as np
from const import DATA_FOLDER

### Load the data
test_data_path = DATA_FOLDER / "test.npz"
test = np.load(test_data_path)
x_test = test["x_test"] # Shape: (42141, 1, 29, 29)
y_test = test["y_test"] # Shape: (42141, 135)

### Reshape x_test from (N, 1, 29, 29) to (N, 29, 29, 1)
# moveaxis(array, source_position, destination_position)
x_test = np.moveaxis(x_test, 1, -1) 

print("Checking adjusted data loading:")
print(f"x_test shape: {x_test.shape}") # Should be (42141, 29, 29, 1)
print(f"y_test shape: {y_test.shape}")


### Evaluate the model
# model.evaluate returns a list: [loss, accuracy]
print("\nEvaluating model on test data...")
results = model.evaluate(x_test, y_test, batch_size=128)

print(f"Test Accuracy: {results[1]:.4f}")
