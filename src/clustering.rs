use anyhow::Result;
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use ndarray::{Array2, Array1};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterResult {
    pub centroids: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub inertia: f64,
}

pub struct KMeansClusterer {
    n_clusters: usize,
    max_iterations: usize,
}

impl KMeansClusterer {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iterations: 300,
        }
    }
    
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }
    
    pub fn fit(&self, data: Vec<Vec<f64>>) -> Result<ClusterResult> {
        let n_samples = data.len();
        let n_features = data[0].len();
        
        // Convert to ndarray
        let flat_data: Vec<f64> = data.into_iter().flatten().collect();
        let array_data = Array2::from_shape_vec((n_samples, n_features), flat_data)?;
        
        // Create and fit KMeans
        let model = KMeans::params(self.n_clusters)
            .max_n_iterations(self.max_iterations)
            .fit(&array_data)?;
        
        // Get predictions
        let labels = model.predict(&array_data);
        
        Ok(ClusterResult {
            centroids: model.centroids().outer_iter()
                .map(|row| row.to_vec())
                .collect(),
            labels: labels.to_vec(),
            inertia: model.iteration_count() as f64, // Simplified
        })
    }
}

// Demo function
pub fn demo_clustering() -> Result<()> {
    // Generate sample data: 2D points in three clusters
    let mut rng = thread_rng();
    let mut data = Vec::new();
    
    // Cluster 1: centered around (1, 1)
    for _ in 0..50 {
        data.push(vec![rng.gen_range(0.5..1.5), rng.gen_range(0.5..1.5)]);
    }
    
    // Cluster 2: centered around (4, 4)
    for _ in 0..50 {
        data.push(vec![rng.gen_range(3.5..4.5), rng.gen_range(3.5..4.5)]);
    }
    
    // Cluster 3: centered around (7, 1)
    for _ in 0..50 {
        data.push(vec![rng.gen_range(6.5..7.5), rng.gen_range(0.5..1.5)]);
    }
    
    let clusterer = KMeansClusterer::new(3);
    let result = clusterer.fit(data)?;
    
    println!("Cluster centroids: {:?}", result.centroids);
    println!("First 10 labels: {:?}", &result.labels[..10]);
    
    Ok(())
}
