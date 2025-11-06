use anyhow::Result;
use rust_stemmers::{Algorithm, Stemmer};
use std::collections::{HashMap, HashSet};
use unicode_segmentation::UnicodeSegmentation;
use whatlang::Lang;

#[derive(Debug)]
pub struct TextProcessor {
    stop_words: HashSet<String>,
    stemmer: Stemmer,
}

impl TextProcessor {
    pub fn new() -> Self {
        let stop_words = Self::get_english_stop_words();
        let stemmer = Stemmer::create(Algorithm::English);
        
        Self { stop_words, stemmer }
    }
    
    fn get_english_stop_words() -> HashSet<String> {
        let words = vec![
            "a", "an", "the", "and", "or", "but", "if", "while", "which", "who",
            "what", "where", "when", "how", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "in", "on", "at", "to", "for", "of", "with", "by", "about", "against",
            "between", "into", "through", "during", "before", "after", "above", "below",
            "from", "up", "down", "out", "off", "over", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can", "just",
        ];
        
        words.into_iter().map(|s| s.to_string()).collect()
    }
    
    pub fn detect_language(&self, text: &str) -> Option<Lang> {
        whatlang::detect_lang(text)
    }
    
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.unicode_words()
            .map(|word| word.to_lowercase())
            .collect()
    }
    
    pub fn remove_stop_words(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .filter(|token| !self.stop_words.contains(token.as_str()))
            .cloned()
            .collect()
    }
    
    pub fn stem(&self, tokens: &[String]) -> Vec<String> {
        tokens
            .iter()
            .map(|token| self.stemmer.stem(token).to_string())
            .collect()
    }
    
    pub fn preprocess_text(&self, text: &str) -> Vec<String> {
        let tokens = self.tokenize(text);
        let without_stops = self.remove_stop_words(&tokens);
        self.stem(&without_stops)
    }
    
    pub fn calculate_tfidf(&self, documents: &[String]) -> HashMap<String, HashMap<usize, f64>> {
        let mut tfidf = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let n_docs = documents.len();
        
        // Preprocess all documents and calculate document frequency
        let processed_docs: Vec<Vec<String>> = documents
            .iter()
            .map(|doc| self.preprocess_text(doc))
            .collect();
        
        // Calculate document frequency for each term
        for doc_terms in &processed_docs {
            let unique_terms: HashSet<String> = doc_terms.iter().cloned().collect();
            for term in unique_terms {
                *doc_freq.entry(term).or_insert(0) += 1;
            }
        }
        
        // Calculate TF-IDF for each term in each document
        for (doc_id, doc_terms) in processed_docs.iter().enumerate() {
            let term_freq = self.calculate_term_frequency(doc_terms);
            
            for (term, tf) in term_freq {
                let df = doc_freq.get(&term).unwrap_or(&1);
                let idf = (n_docs as f64 / *df as f64).ln();
                let tfidf_score = tf * idf;
                
                tfidf.entry(term)
                    .or_insert_with(HashMap::new)
                    .insert(doc_id, tfidf_score);
            }
        }
        
        tfidf
    }
    
    fn calculate_term_frequency(&self, terms: &[String]) -> HashMap<String, f64> {
        let mut tf = HashMap::new();
        let total_terms = terms.len() as f64;
        
        for term in terms {
            *tf.entry(term.clone()).or_insert(0.0) += 1.0;
        }
        
        // Normalize
        for count in tf.values_mut() {
            *count /= total_terms;
        }
        
        tf
    }
    
    pub fn cosine_similarity(&self, vec1: &HashMap<String, f64>, vec2: &HashMap<String, f64>) -> f64 {
        let dot_product: f64 = vec1.iter()
            .filter_map(|(term, &score1)| vec2.get(term).map(|&score2| score1 * score2))
            .sum();
        
        let norm1: f64 = vec1.values().map(|&score| score * score).sum::<f64>().sqrt();
        let norm2: f64 = vec2.values().map(|&score| score * score).sum::<f64>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
}

// Demo NLP functionality
pub fn demo_nlp() -> Result<()> {
    let processor = TextProcessor::new();
    
    let documents = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Machine learning is a subset of artificial intelligence".to_string(),
        "Rust is a systems programming language that runs blazingly fast".to_string(),
        "Natural language processing helps computers understand human language".to_string(),
    ];
    
    // Detect language
    for doc in &documents {
        if let Some(lang) = processor.detect_language(doc) {
            println!("Document: '{}' -> Language: {:?}", doc, lang);
        }
    }
    
    // Preprocess text
    let sample_text = "The quick brown foxes are jumping over lazy dogs";
    let processed = processor.preprocess_text(sample_text);
    println!("Processed text: {:?}", processed);
    
    // Calculate TF-IDF
    let tfidf = processor.calculate_tfidf(&documents);
    println!("TF-IDF features calculated for {} terms", tfidf.len());
    
    Ok(())
}
