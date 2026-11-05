import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully!")
print(type(model))
print(model)