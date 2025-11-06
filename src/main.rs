mod linear_regression;
mod clustering;
mod neural_network;
mod nlp;
mod decision_tree;
mod web_api;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rust-ai")]
#[command(about = "AI/ML Toolkit in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run linear regression demo
    LinearRegression,
    /// Run clustering demo  
    Clustering,
    /// Run neural network demo
    NeuralNetwork,
    /// Run NLP demo
    Nlp,
    /// Run decision tree demo
    DecisionTree,
    /// Start AI web API
    Server,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::LinearRegression => {
            println!("Running Linear Regression Demo...");
            linear_regression::demo_linear_regression()?;
        }
        Commands::Clustering => {
            println!("Running Clustering Demo...");
            clustering::demo_clustering()?;
        }
        Commands::NeuralNetwork => {
            println!("Running Neural Network Demo...");
            neural_network::demo_neural_network()?;
        }
        Commands::Nlp => {
            println!("Running NLP Demo...");
            nlp::demo_nlp()?;
        }
        Commands::DecisionTree => {
            println!("Running Decision Tree Demo...");
            decision_tree::demo_decision_tree()?;
        }
        Commands::Server => {
            println!("Starting AI Web Server...");
            web_api::start_ai_server().await?;
        }
    }
    
    Ok(())
}
