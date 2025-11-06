use anyhow::Result;
use linfa::traits::{Fit, Predict};
use linfa_trees::DecisionTree;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionTreeModel {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
}

pub struct DecisionTreeClassifier {
    model: Option<DecisionTree<f64, usize>>,
    config: DecisionTreeModel,
}

impl DecisionTreeClassifier {
    pub fn new() -> Self {
        Self {
            model: None,
            config: DecisionTreeModel {
                max_depth: Some(5),
                min_samples_split: 2,
            },
        }
    }
    
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = Some(depth);
        self
    }
    
    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.config.min_samples_split = min_samples;
        self
    }
    
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<usize>) -> Result<()> {
        let x_array = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())?;
        let y_array = Array1::from_vec(y);
        
        let dataset = linfa::Dataset::new(x_array, y_array);
        
        let model = DecisionTree::params()
            .max_depth(self.config.max_depth)
            .min_samples_split(self.config.min_samples_split)
            .fit(&dataset)?;
        
        self.model = Some(model);
        Ok(())
    }
    
    pub fn predict(&self, x: Vec<Vec<f64>>) -> Result<Vec<usize>> {
        let model = self.model.as_ref().ok_or_else(|| anyhow::anyhow!("Model not trained"))?;
        
        let x_array = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())?;
        let predictions = model.predict(&x_array);
        
        Ok(predictions.to_vec())
    }
    
    pub fn feature_importance(&self) -> Result<Vec<f64>> {
        let model = self.model.as_ref().ok_or_else(|| anyhow::anyhow!("Model not trained"))?;
        
        // In real implementation, you'd extract feature importance from the model
        // This is a simplified version
        Ok(vec![1.0; 4]) // Placeholder
    }
}

// Demo function
pub fn demo_decision_tree() -> Result<()> {
    let mut model = DecisionTreeClassifier::new()
        .with_max_depth(3)
        .with_min_samples_split(2);
    
    // Iris dataset-like example
    let x_train = vec![
        vec![5.1, 3.5, 1.4, 0.2],  // Setosa
        vec![4.9, 3.0, 1.4, 0.2],  // Setosa
        vec![7.0, 3.2, 4.7, 1.4],  // Versicolor
        vec![6.4, 3.2, 4.5, 1.5],  // Versicolor
        vec![6.3, 3.3, 6.0, 2.5],  // Virginica
        vec![5.8, 2.7, 5.1, 1.9],  // Virginica
    ];
    
    let y_train = vec![0, 0, 1, 1, 2, 2]; // Class labels
    
    model.fit(x_train, y_train)?;
    
    // Test predictions
    let x_test = vec![
        vec![5.0, 3.6, 1.3, 0.25],  // Should be Setosa
        vec![6.5, 3.0, 4.6, 1.5],   // Should be Versicolor
    ];
    
    let predictions = model.predict(x_test)?;
    println!("Predictions: {:?}", predictions);
    
    Ok(())
}
