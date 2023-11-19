from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# Assume you have preprocessed and loaded your data into all_images_normalized and all_labels_one_hot

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(all_images_normalized, all_labels_one_hot, test_size=0.2, random_state=42)

# Function to create a CNN model
def create_model(optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=activation, input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dense(2, activation='softmax'))  # Assuming 2 classes (benign and malignant)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier based on the create_model function
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Define the hyperparameter grid
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh', 'sigmoid']
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid_search.fit(X_train, y_train)

# Print the best parameters and corresponding accuracy
print(f"Best Parameters: {grid_result.best_params_}")
print(f"Best Accuracy: {grid_result.best_score_}")

# Evaluate the best model on the validation set
best_model = grid_result.best_estimator_
val_accuracy = best_model.score(X_val, y_val)
print(f"Validation Accuracy with Best Model: {val_accuracy}")
