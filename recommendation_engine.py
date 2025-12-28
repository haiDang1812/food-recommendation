import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

from utils import parse_nutrition, clean_ingredients, clean_tags


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing.
    """

    def __init__(self, file_path):
        """
        Initialize preprocessor with data file path.

        Args:
            file_path: Path to CSV file
        """
        self.file_path = file_path
        self.df = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load data from CSV file."""
        self.df = pd.read_csv(self.file_path)
        return self.df

    def handle_missing_values(self):
        """Handle missing values in dataset."""
        # Drop rows with missing critical fields
        self.df = self.df.dropna(subset=['name', 'ingredients', 'nutrition'])

        # Fill missing descriptions
        self.df['description'] = self.df['description'].fillna('')

        # Fill missing tags
        self.df['tags'] = self.df['tags'].fillna('[]')

        return self.df

    def remove_duplicates(self):
        """Remove duplicate recipes."""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['name'], keep='first')
        removed = initial_count - len(self.df)
        print(f"Removed {removed} duplicate recipes")
        return self.df

    def remove_outliers(self):
        """Remove outlier recipes based on cooking time and calories."""
        initial_count = len(self.df)

        # Remove extreme cooking times
        self.df = self.df[self.df['minutes'] <= 1000]

        # Parse nutrition before filtering
        self._parse_nutrition_values()

        # Remove extreme calories
        self.df = self.df[self.df['calories'] <= 5000]

        removed = initial_count - len(self.df)
        print(f"Removed {removed} outlier recipes")
        return self.df

    def _parse_nutrition_values(self):
        """Parse nutrition column into individual columns."""
        nutrition_data = self.df['nutrition'].apply(parse_nutrition)
        nutrition_df = pd.DataFrame(nutrition_data.tolist())
        self.df = pd.concat([self.df, nutrition_df], axis=1)
        return self.df

    def clean_text_fields(self):
        """Clean and prepare text fields for vectorization."""
        # Clean ingredients
        self.df['ingredients_clean'] = self.df['ingredients'].apply(clean_ingredients)

        # Clean tags
        self.df['tags_clean'] = self.df['tags'].apply(clean_tags)

        # Combine for content-based filtering
        self.df['content'] = self.df['ingredients_clean'] + ' ' + self.df['tags_clean']

        return self.df

    def normalize_features(self):
        """Normalize numerical features using StandardScaler."""
        numerical_cols = ['minutes', 'calories', 'fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbs']

        # Handle any remaining NaN values
        self.df[numerical_cols] = self.df[numerical_cols].fillna(0)

        # Save original values before normalization
        for col in numerical_cols:
            self.df[f'{col}_original'] = self.df[col]

        # Normalize (optional - comment out if not needed for content-based filtering)
        # self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])

        return self.df

    def preprocess_all(self):
        """
        Execute full preprocessing pipeline.

        Returns:
            Preprocessed DataFrame
        """
        print("Loading data...")
        self.load_data()
        print(f"Initial dataset size: {len(self.df)} recipes")

        print("\nHandling missing values...")
        self.handle_missing_values()

        print("Removing duplicates...")
        self.remove_duplicates()

        print("Removing outliers...")
        self.remove_outliers()

        print("Cleaning text fields...")
        self.clean_text_fields()

        print("Normalizing features...")
        self.normalize_features()

        print("Resetting index...")
        self.df = self.df.reset_index(drop=True)

        print(f"\nFinal dataset size: {len(self.df)} recipes")
        return self.df


class TFIDFRecommender:
    """
    Content-based recommender using TF-IDF vectorization.
    """

    def __init__(self, df, max_features=500):
        """
        Initialize TF-IDF recommender.

        Args:
            df: Preprocessed DataFrame
            max_features: Maximum features for TF-IDF
        """
        self.df = df
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def fit(self):
        """Fit TF-IDF vectorizer."""
        print("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['content'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        return self

    def recommend_by_recipe(self, recipe_name, n_recommendations=10):
        """
        Get recommendations based on a recipe name.

        Args:
            recipe_name: Name of base recipe
            n_recommendations: Number of recommendations

        Returns:
            DataFrame with recommendations
        """
        # Find recipe index
        matches = self.df[self.df['name'].str.contains(recipe_name, case=False, na=False)]
        if len(matches) == 0:
            return pd.DataFrame()

        idx = matches.index[0]

        # Compute similarity on-demand (only for this recipe)
        recipe_vector = self.tfidf_matrix[idx]
        similarity_scores = cosine_similarity(recipe_vector, self.tfidf_matrix).flatten()

        # Get top recommendations (excluding the recipe itself)
        sorted_indices = similarity_scores.argsort()[::-1]
        top_indices = [i for i in sorted_indices[1:n_recommendations+1]]
        top_scores = [similarity_scores[i] for i in top_indices]

        # Create recommendations DataFrame
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = top_scores

        return recommendations

    def recommend_by_ingredients(self, ingredients_text, n_recommendations=10):
        """
        Get recommendations based on available ingredients.

        Args:
            ingredients_text: Space-separated ingredients
            n_recommendations: Number of recommendations

        Returns:
            DataFrame with recommendations
        """
        # Transform input ingredients
        input_vector = self.vectorizer.transform([ingredients_text])

        # Calculate similarity with all recipes
        similarity_scores = cosine_similarity(input_vector, self.tfidf_matrix)[0]

        # Get top recommendations
        top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
        top_scores = similarity_scores[top_indices]

        # Create recommendations DataFrame
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = top_scores

        return recommendations

    def save_model(self, path='models/tfidf_model.pkl'):
        """Save TF-IDF model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)
        print(f"TF-IDF model saved to {path}")

    def load_model(self, path='models/tfidf_model.pkl'):
        """Load TF-IDF model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
        print(f"TF-IDF model loaded from {path}")


class AdvancedEmbeddingRecommender:
    """
    Content-based recommender using advanced embeddings (Word2Vec or Sentence-BERT).
    """

    def __init__(self, df, method='word2vec'):
        """
        Initialize advanced embedding recommender.

        Args:
            df: Preprocessed DataFrame
            method: 'word2vec' or 'sentence-bert'
        """
        self.df = df
        self.method = method
        self.embeddings = None
        self.similarity_matrix = None
        self.model = None

    def fit(self):
        """Fit embedding model."""
        if self.method == 'word2vec':
            self._fit_word2vec()
        elif self.method == 'sentence-bert':
            self._fit_sentence_bert()

        return self

    def _fit_word2vec(self):
        """Fit Word2Vec model."""
        print("Training Word2Vec model...")

        # Tokenize content
        tokenized_content = [text.split() for text in self.df['content']]

        # Train Word2Vec
        self.model = Word2Vec(
            sentences=tokenized_content,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            epochs=10
        )

        # Generate embeddings by averaging word vectors
        embeddings_list = []
        for tokens in tokenized_content:
            word_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if word_vectors:
                embeddings_list.append(np.mean(word_vectors, axis=0))
            else:
                embeddings_list.append(np.zeros(100))

        self.embeddings = np.array(embeddings_list)
        print(f"Word2Vec embeddings shape: {self.embeddings.shape}")

    def _fit_sentence_bert(self):
        """Fit Sentence-BERT model."""
        from sentence_transformers import SentenceTransformer

        print("Loading Sentence-BERT model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Generating embeddings...")
        # Combine ingredients and description for richer embeddings
        texts = self.df['ingredients_clean'] + ' ' + self.df['description'].fillna('')
        self.embeddings = self.model.encode(texts.tolist(), show_progress_bar=True)
        print(f"Sentence-BERT embeddings shape: {self.embeddings.shape}")

    def recommend_by_recipe(self, recipe_name, n_recommendations=10):
        """
        Get recommendations based on a recipe name.

        Args:
            recipe_name: Name of base recipe
            n_recommendations: Number of recommendations

        Returns:
            DataFrame with recommendations
        """
        # Find recipe index
        matches = self.df[self.df['name'].str.contains(recipe_name, case=False, na=False)]
        if len(matches) == 0:
            return pd.DataFrame()

        idx = matches.index[0]

        # Compute similarity on-demand (only for this recipe)
        recipe_vector = self.embeddings[idx].reshape(1, -1)
        similarity_scores = cosine_similarity(recipe_vector, self.embeddings).flatten()

        # Get top recommendations (excluding the recipe itself)
        sorted_indices = similarity_scores.argsort()[::-1]
        top_indices = [i for i in sorted_indices[1:n_recommendations+1]]
        top_scores = [similarity_scores[i] for i in top_indices]

        # Create recommendations DataFrame
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = top_scores

        return recommendations

    def recommend_by_ingredients(self, ingredients_text, n_recommendations=10):
        """
        Get recommendations based on available ingredients.

        Args:
            ingredients_text: Space-separated ingredients
            n_recommendations: Number of recommendations

        Returns:
            DataFrame with recommendations
        """
        # Generate embedding for input
        if self.method == 'word2vec':
            tokens = ingredients_text.split()
            word_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if word_vectors:
                input_vector = np.mean(word_vectors, axis=0).reshape(1, -1)
            else:
                input_vector = np.zeros((1, 100))
        else:
            input_vector = self.model.encode([ingredients_text])

        # Calculate similarity
        similarity_scores = cosine_similarity(input_vector, self.embeddings)[0]

        # Get top recommendations
        top_indices = similarity_scores.argsort()[-n_recommendations:][::-1]
        top_scores = similarity_scores[top_indices]

        # Create recommendations DataFrame
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = top_scores

        return recommendations

    def save_model(self, path='models/advanced_embeddings.pkl'):
        """Save embeddings to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'embeddings': self.embeddings,
                'model': self.model
            }, f)
        print(f"Advanced embeddings saved to {path}")

    def load_model(self, path='models/advanced_embeddings.pkl'):
        """Load embeddings from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.method = data['method']
            self.embeddings = data['embeddings']
            self.model = data.get('model')
        print(f"Advanced embeddings loaded from {path}")


class EvaluationMetrics:
    """
    Calculate evaluation metrics for recommendation systems.
    """

    @staticmethod
    def calculate_diversity(recommendations_df):
        """
        Calculate diversity score based on unique tags.

        Args:
            recommendations_df: DataFrame with recommendations

        Returns:
            float: Diversity score (0-1)
        """
        all_tags = []
        for tags_str in recommendations_df['tags']:
            try:
                import ast
                tags_list = ast.literal_eval(tags_str)
                all_tags.extend(tags_list)
            except:
                pass

        if len(all_tags) == 0:
            return 0.0

        unique_tags = len(set(all_tags))
        total_tags = len(all_tags)
        diversity = unique_tags / total_tags

        return diversity

    @staticmethod
    def calculate_coverage(recommendations_df, full_df):
        """
        Calculate coverage metric.

        Args:
            recommendations_df: DataFrame with recommendations
            full_df: Full dataset DataFrame

        Returns:
            float: Coverage percentage
        """
        recommended_ids = set(recommendations_df.index)
        total_items = len(full_df)
        coverage = len(recommended_ids) / total_items * 100

        return coverage

    @staticmethod
    def calculate_precision_at_k(base_recipe, recommendations_df, k=10, min_common_tags=3):
        """
        Calculate Precision@K based on tag overlap.

        Args:
            base_recipe: Base recipe row
            recommendations_df: DataFrame with recommendations
            k: Number of top recommendations to consider
            min_common_tags: Minimum common tags to be considered relevant

        Returns:
            float: Precision@K score
        """
        import ast

        try:
            base_tags = set(ast.literal_eval(base_recipe['tags']))
        except:
            return 0.0

        relevant_count = 0
        top_k_recs = recommendations_df.head(k)

        for _, rec in top_k_recs.iterrows():
            try:
                rec_tags = set(ast.literal_eval(rec['tags']))
                common_tags = len(base_tags.intersection(rec_tags))
                if common_tags >= min_common_tags:
                    relevant_count += 1
            except:
                pass

        precision = relevant_count / k if k > 0 else 0
        return precision

    @staticmethod
    def calculate_recall_at_k(base_recipe, recommendations_df, full_df, k=10, min_common_tags=3):
        """
        Calculate Recall@K based on tag overlap.

        Args:
            base_recipe: Base recipe row
            recommendations_df: DataFrame with recommendations
            full_df: Full dataset
            k: Number of top recommendations to consider
            min_common_tags: Minimum common tags to be considered relevant

        Returns:
            float: Recall@K score
        """
        import ast

        try:
            base_tags = set(ast.literal_eval(base_recipe['tags']))
        except:
            return 0.0

        # Find all relevant recipes in dataset
        relevant_recipes = []
        for _, recipe in full_df.iterrows():
            try:
                rec_tags = set(ast.literal_eval(recipe['tags']))
                common_tags = len(base_tags.intersection(rec_tags))
                if common_tags >= min_common_tags and recipe['name'] != base_recipe['name']:
                    relevant_recipes.append(recipe['name'])
            except:
                pass

        if len(relevant_recipes) == 0:
            return 0.0

        # Count relevant recipes in top K recommendations
        top_k_recs = recommendations_df.head(k)
        recommended_names = set(top_k_recs['name'].tolist())
        relevant_in_top_k = len(set(relevant_recipes).intersection(recommended_names))

        recall = relevant_in_top_k / len(relevant_recipes)
        return recall

    @staticmethod
    def compare_methods(tfidf_recommender, advanced_recommender, test_recipes, n_recommendations=10):
        """
        Compare TF-IDF vs Advanced Embeddings.

        Args:
            tfidf_recommender: TFIDFRecommender instance
            advanced_recommender: AdvancedEmbeddingRecommender instance
            test_recipes: List of recipe names to test
            n_recommendations: Number of recommendations per recipe

        Returns:
            dict: Comparison metrics
        """
        tfidf_diversity = []
        advanced_diversity = []

        for recipe in test_recipes:
            # TF-IDF recommendations
            tfidf_recs = tfidf_recommender.recommend_by_recipe(recipe, n_recommendations)
            if len(tfidf_recs) > 0:
                tfidf_diversity.append(EvaluationMetrics.calculate_diversity(tfidf_recs))

            # Advanced recommendations
            advanced_recs = advanced_recommender.recommend_by_recipe(recipe, n_recommendations)
            if len(advanced_recs) > 0:
                advanced_diversity.append(EvaluationMetrics.calculate_diversity(advanced_recs))

        results = {
            'tfidf_avg_diversity': np.mean(tfidf_diversity) if tfidf_diversity else 0,
            'advanced_avg_diversity': np.mean(advanced_diversity) if advanced_diversity else 0,
            'tfidf_diversity_std': np.std(tfidf_diversity) if tfidf_diversity else 0,
            'advanced_diversity_std': np.std(advanced_diversity) if advanced_diversity else 0
        }

        return results
