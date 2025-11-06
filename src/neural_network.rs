use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap, Adam};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetworkConfig {
    pub input_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub output_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
}

struct SimpleNN {
    layers: Vec<Linear>,
}

impl SimpleNN {
    fn new(vs: VarBuilder, config: &NeuralNetworkConfig) -> Result<Self> {
        let mut layers = Vec::new();
        let mut input_size = config.input_size;
        
        // Hidden layers
        for &hidden_size in &config.hidden_sizes {
            let linear = candle_nn::linear(input_size, hidden_size, vs.pp("linear"))?;
            layers.push(linear);
            input_size = hidden_size;
        }
        
        // Output layer
        let output_layer = candle_nn::linear(input_size, config.output_size, vs.pp("output"))?;
        layers.push(output_layer);
        
        Ok(Self { layers })
    }
    
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs)?;
            if i < self.layers.len() - 1 {
                // ReLU activation for hidden layers
                xs = ops::relu(&xs)?;
            }
        }
        
        Ok(xs)
    }
}

pub struct NeuralNetwork {
    config: NeuralNetworkConfig,
    varmap: VarMap,
}

impl NeuralNetwork {
    pub fn new(config: NeuralNetworkConfig) -> Self {
        Self {
            config,
            varmap: VarMap::new(),
        }
    }
    
    pub fn train(&mut self, x_train: &[f64], y_train: &[f64]) -> Result<Vec<f64>> {
        let device = Device::Cpu;
        let vs = VarBuilder::from_varmap(&self.varmap, candle_core::DType::F64, &device);
        
        let model = SimpleNN::new(vs, &self.config)?;
        let mut adam = Adam::new(
            self.varmap.all_vars(),
            self.config.learning_rate,
        )?;
        
        let n_samples = y_train.len();
        let x_tensor = Tensor::from_vec(
            x_train.to_vec(),
            (n_samples, self.config.input_size),
            &device,
        )?;
        
        let y_tensor = Tensor::from_vec(
            y_train.to_vec(),
            (n_samples, self.config.output_size),
            &device,
        )?;
        
        let mut losses = Vec::new();
        
        for epoch in 0..self.config.epochs {
            let logits = model.forward(&x_tensor)?;
            let loss = loss::mse(&logits, &y_tensor)?;
            
            adam.backward_step(&loss)?;
            
            let loss_value = loss.to_vec0::<f64>()?;
            losses.push(loss_value);
            
            if epoch % 100 == 0 {
                println!("Epoch {}: loss = {:.4}", epoch, loss_value);
            }
        }
        
        Ok(losses)
    }
    
    pub fn predict(&self, x: &[f64]) -> Result<Vec<f64>> {
        let device = Device::Cpu;
        let vs = VarBuilder::from_varmap(&self.varmap, candle_core::DType::F64, &device);
        
        let model = SimpleNN::new(vs, &self.config)?;
        
        let x_tensor = Tensor::from_vec(
            x.to_vec(),
            (1, self.config.input_size),
            &device,
        )?;
        
        let prediction = model.forward(&x_tensor)?;
        Ok(prediction.to_vec1()?)
    }
}

// Demo XOR problem
pub fn demo_neural_network() -> Result<()> {
    let config = NeuralNetworkConfig {
        input_size: 2,
        hidden_sizes: vec![4, 4],
        output_size: 1,
        learning_rate: 0.01,
        epochs: 1000,
    };
    
    let mut model = NeuralNetwork::new(config);
    
    // XOR dataset
    let x_train = vec![
        0.0, 0.0,
        0.0, 1.0, 
        1.0, 0.0,
        1.0, 1.0,
    ];
    
    let y_train = vec![
        0.0,  // 0 XOR 0 = 0
        1.0,  // 0 XOR 1 = 1
        1.0,  // 1 XOR 0 = 1  
        0.0,  // 1 XOR 1 = 0
    ];
    
    let losses = model.train(&x_train, &y_train)?;
    
    // Test predictions
    for (i, (x1, x2)) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)].iter().enumerate() {
        let input = vec![*x1, *x2];
        let prediction = model.predict(&input)?;
        println!("Input: ({}, {}) -> Prediction: {:.4}, Expected: {}", 
                 x1, x2, prediction[0], y_train[i]);
    }
    
    Ok(())
}
