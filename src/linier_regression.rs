use anyhow::Result;
use linfa::traits::{Fit, Predict};
use linfa_linear::LinearRegression;
use ndarray::{Array, Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct LinearModel {
    intercept: f64,
    coefficients: Vec<f64>,
}

pub struct SimpleLinearRegression {
    model: Option<LinearRegression<f64, ndarray::Ix1>>,
}

impl SimpleLinearRegression {
    pub fn new() -> Self {
        Self { model: None }
    }
    
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<LinearModel> {
        let x_array = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())?;
        let y_array = Array1::from_vec(y);
        
        let dataset = linfa::Dataset::new(x_array, y_array);
        
        let model = LinearRegression::default().fit(&dataset)?;
        let parameters = model.parameters();
        
        let linear_model = LinearModel {
            intercept: parameters.intercept,
            coefficients: parameters.coefficients.to_vec(),
        };
        
        self.model = Some(model);
        Ok(linear_model)
    }
    
    pub fn predict(&self, x: Vec<Vec<f64>>) -> Result<Vec<f64>> {
        let model = self.model.as_ref().ok_or_else(|| anyhow::anyhow!("Model not trained"))?;
        
        let x_array = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())?;
        let predictions = model.predict(&x_array);
        
        Ok(predictions.to_vec())
    }
}

// Example usage
pub fn demo_linear_regression() -> Result<()> {
    let mut model = SimpleLinearRegression::new();
    
    // Training data: [square_meters, num_bedrooms] -> price
    let x_train = vec![
        vec![50.0, 1.0],
        vec![75.0, 2.0],
        vec![100.0, 3.0],
        vec![120.0, 3.0],
        vec![150.0, 4.0],
    ];
    
    let y_train = vec![
        100000.0,  // $100,000
        150000.0,  // $150,000  
        200000.0,  // $200,000
        250000.0,  // $250,000
        300000.0,  // $300,000
    ];
    
    let trained_model = model.fit(x_train, y_train)?;
    println!("Trained model: {:?}", trained_model);
    
    // Predict
    let x_test = vec![vec![80.0, 2.0], vec![130.0, 3.0]];
    let predictions = model.predict(x_test)?;
    
    println!("Predictions: {:?}", predictions);
    
    Ok(())
}
