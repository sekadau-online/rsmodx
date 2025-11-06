# rsmodx
machine learning simple

# Linear Regression
cargo run -- linear-regression

# Clustering  
cargo run -- clustering

# Neural Network
cargo run -- neural-network

# NLP
cargo run -- nlp

# Decision Tree
cargo run -- decision-tree

# Web API
cargo run -- server

curl -X POST http://localhost:3030/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
