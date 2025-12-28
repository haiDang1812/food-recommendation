import streamlit as st
import pandas as pd
import numpy as np
import os
import ast
from recommendation_engine import DataPreprocessor, TFIDFRecommender, AdvancedEmbeddingRecommender, EvaluationMetrics
from utils import create_visualizations, filter_by_context, explain_recommendation, plot_embeddings_tsne

# Page configuration
st.set_page_config(
    page_title="Food Recommendation System",
    page_icon="🍽️",
    layout="wide"
)


# Data loading and caching
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess recipe data."""
    preprocessor = DataPreprocessor('data/recipes_sampled.csv')
    df = preprocessor.preprocess_all()
    return df, preprocessor


@st.cache_data
def get_unique_ingredients(_df):
    """Extract unique ingredients from dataset for autocomplete."""
    all_ingredients = []
    for ing_str in _df['ingredients']:
        try:
            ing_list = ast.literal_eval(ing_str)
            all_ingredients.extend([ing.strip().lower() for ing in ing_list])
        except:
            pass

    # Get unique and sort
    unique_ingredients = sorted(list(set(all_ingredients)))
    return unique_ingredients


@st.cache_resource
def load_tfidf_recommender(_df):
    """Load or create TF-IDF recommender."""
    recommender = TFIDFRecommender(_df, max_features=500)

    if os.path.exists('models/tfidf_model.pkl'):
        print("✓ Loading existing TF-IDF model from disk...")
        recommender.load_model()
    else:
        print("✗ Training new TF-IDF model...")
        with st.spinner("Training TF-IDF model..."):
            recommender.fit()
            recommender.save_model()

    return recommender


@st.cache_resource
def load_advanced_recommender(_df, method='word2vec'):
    """Load or create advanced embedding recommender."""
    model_path = f'models/{method}_embeddings.pkl'
    recommender = AdvancedEmbeddingRecommender(_df, method=method)

    if os.path.exists(model_path):
        print(f"✓ Loading existing {method} model from disk...")
        recommender.load_model(model_path)
    else:
        print(f"✗ Training new {method} model...")
        with st.spinner(f"Training {method.upper()} model..."):
            recommender.fit()
            recommender.save_model(model_path)

    return recommender


# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if 'liked_recipes' not in st.session_state:
        st.session_state.liked_recipes = []
    if 'viewed_recipes' not in st.session_state:
        st.session_state.viewed_recipes = []


# Main application
def main():
    initialize_session_state()

    # Header
    st.title("🍽️ Food Recommendation System")
    st.markdown("Discover delicious recipes tailored to your taste and dietary needs")

    # Load data
    try:
        df, preprocessor = load_and_preprocess_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home & Overview", "🍽️ Discover Food", "📊 Model Performance", "📖 My History"])

    # Tab 1: Home & Overview
    with tab1:
        st.header("Dataset Overview")

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Recipes", f"{len(df):,}")
        with col2:
            avg_time = df['minutes'].mean()
            st.metric("Avg Cooking Time", f"{avg_time:.0f} min")
        with col3:
            avg_cal = df['calories'].mean()
            st.metric("Avg Calories", f"{avg_cal:.0f}")
        with col4:
            avg_protein = df['protein'].mean()
            st.metric("Avg Protein", f"{avg_protein:.1f}g")

        st.divider()

        # Data Preprocessing Visualizations
        st.subheader("📊 Data Preprocessing Analysis")

        try:
            # Load original data for comparison
            df_original = pd.read_csv('data/recipes_sampled.csv')

            # Parse nutrition column to get calories for original data
            from utils import parse_nutrition
            nutrition_data = df_original['nutrition'].apply(parse_nutrition)
            for col in ['calories', 'fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbs']:
                df_original[col] = nutrition_data.apply(lambda x: x[col])

            # 1. Missing Values Bar Chart
            st.markdown("#### Missing Values Analysis")
            import plotly.graph_objects as go

            # Calculate missing values before preprocessing
            missing_before = df_original[['name', 'minutes', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients']].isnull().sum()
            missing_after = df[['name', 'minutes', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients']].isnull().sum()

            fig_missing = go.Figure(data=[
                go.Bar(name='Before Cleaning', x=missing_before.index, y=missing_before.values, marker_color='#FF6B6B'),
                go.Bar(name='After Cleaning', x=missing_after.index, y=missing_after.values, marker_color='#4ECDC4')
            ])
            fig_missing.update_layout(
                title='Missing Values: Before vs After Cleaning',
                xaxis_title='Columns',
                yaxis_title='Number of Missing Values',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_missing, use_container_width=True)

            st.divider()

            # 2. Outlier Detection - Box Plots
            st.markdown("#### Outlier Detection & Removal")

            col1, col2 = st.columns(2)

            with col1:
                # Minutes boxplot
                fig_minutes = go.Figure()
                fig_minutes.add_trace(go.Box(y=df_original['minutes'], name='Before', marker_color='#FF6B6B'))
                fig_minutes.add_trace(go.Box(y=df['minutes'], name='After', marker_color='#4ECDC4'))
                fig_minutes.update_layout(
                    title='Cooking Time Distribution (minutes)',
                    yaxis_title='Minutes',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_minutes, use_container_width=True)

            with col2:
                # Calories boxplot
                fig_calories = go.Figure()
                fig_calories.add_trace(go.Box(y=df_original['calories'], name='Before', marker_color='#FF6B6B'))
                fig_calories.add_trace(go.Box(y=df['calories'], name='After', marker_color='#4ECDC4'))
                fig_calories.update_layout(
                    title='Calories Distribution',
                    yaxis_title='Calories',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_calories, use_container_width=True)

            st.caption(f"Removed {len(df_original) - len(df)} outlier recipes ({((len(df_original) - len(df)) / len(df_original) * 100):.1f}%)")

            st.divider()

            # 3. Dish Type Distribution - Pie Chart
            st.markdown("#### Dish Type Distribution")

            # Extract dish types from tags
            dish_types = {
                'Main Course': 0,
                'Dessert': 0,
                'Breakfast': 0,
                'Side Dish': 0,
                'Snack': 0,
                'Salad': 0,
                'Other': 0
            }

            for tags_str in df['tags_clean']:
                tags_lower = str(tags_str).lower()
                if any(keyword in tags_lower for keyword in ['main-dish', 'lunch', 'dinner']):
                    dish_types['Main Course'] += 1
                elif 'dessert' in tags_lower or 'sweet' in tags_lower:
                    dish_types['Dessert'] += 1
                elif 'breakfast' in tags_lower or 'brunch' in tags_lower:
                    dish_types['Breakfast'] += 1
                elif 'side-dish' in tags_lower:
                    dish_types['Side Dish'] += 1
                elif 'snack' in tags_lower or 'appetizer' in tags_lower:
                    dish_types['Snack'] += 1
                elif 'salad' in tags_lower:
                    dish_types['Salad'] += 1
                else:
                    dish_types['Other'] += 1

            fig_pie = go.Figure(data=[go.Pie(
                labels=list(dish_types.keys()),
                values=list(dish_types.values()),
                hole=0.3,
                marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#B8B8B8'])
            )])
            fig_pie.update_layout(
                title='Distribution of Dish Types in Dataset',
                height=500
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            # 4. Sample Data - Content Field Cleaning
            st.markdown("#### Content Field: Before vs After Cleaning")
            st.caption("Example showing how ingredients, tags, and steps are combined into the 'content' field")

            # Show a sample recipe
            sample_idx = 0
            sample_original = df_original.iloc[sample_idx]
            sample_cleaned = df.iloc[sample_idx]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Before (Separate Fields)**")
                st.write(f"**Name:** {sample_original['name']}")
                st.write(f"**Tags:** {sample_original['tags'][:100]}...")
                st.write(f"**Ingredients:** {str(sample_original['ingredients'])[:100]}...")

            with col2:
                st.markdown("**After (Combined Content)**")
                st.write(f"**Name:** {sample_cleaned['name']}")
                st.write(f"**Content:** {str(sample_cleaned['content'])[:200]}...")

            # Show statistics
            st.info(f"📊 Dataset reduced from {len(df_original):,} to {len(df):,} recipes after cleaning ({(len(df)/len(df_original)*100):.1f}% retained)")

        except Exception as e:
            st.warning(f"Could not load preprocessing visualizations: {e}")

        st.divider()

        # Visualizations
        st.subheader("Data Insights")

        try:
            with st.spinner("Creating visualizations..."):
                figs = create_visualizations(df)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(figs['cooking_time'], use_container_width=True)
                st.plotly_chart(figs['nutrition_scatter'], use_container_width=True)

            with col2:
                st.plotly_chart(figs['top_tags'], use_container_width=True)
                st.plotly_chart(figs['correlation'], use_container_width=True)

        except Exception as e:
            st.error(f"Error creating visualizations: {e}")

    # Tab 2: Discover Food
    with tab2:
        st.header("🍽️ Discover Your Perfect Food")

        # Method selection (compact horizontal)
        st.subheader("⚙️ Recommendation Method")
        embedding_method = st.radio(
            "Select embedding method:",
            ["TF-IDF", "Word2Vec"],
            horizontal=True,
            help="Choose the AI method for food recommendations"
        )

        st.divider()

        # Preferences (4 core selections in 2 columns x 2 rows)
        st.subheader("🔧 Set Your Preferences")

        pcol1, pcol2 = st.columns(2)

        with pcol1:
            dish_type = st.selectbox(
                "🍽️ Dish Type",
                ["Any", "Main Course", "Dessert", "Breakfast", "Side Dish", "Snack", "Salad"]
            )

            cooking_time = st.selectbox(
                "⏱️ Cooking Time",
                ["Any", "Quick (<30 min)", "Medium (30-60 min)", "Long (>60 min)"]
            )

        with pcol2:
            dietary_restriction = st.selectbox(
                "🥗 Dietary Preference",
                ["Any", "Vegetarian", "Vegan", "Low Fat", "Healthy"]
            )

            calorie_level = st.selectbox(
                "🔥 Calorie Level",
                ["Any", "Low (<400 cal)", "Medium (400-600 cal)", "High (>600 cal)"]
            )

        st.divider()

        # Available Ingredients (autocomplete multiselect)
        st.subheader("🧺 Available Ingredients (Optional)")
        st.caption("Select ingredients you have. Type to search (e.g., 'chic' → chicken)")

        # Get unique ingredients
        ingredient_list = get_unique_ingredients(df)

        selected_ingredients = st.multiselect(
            "Search and select ingredients:",
            options=ingredient_list,
            placeholder="Start typing to search ingredients..."
        )

        if selected_ingredients:
            st.success(f"Selected: {', '.join(selected_ingredients)}")

        st.divider()

        # Load recommender
        if embedding_method == "TF-IDF":
            recommender = load_tfidf_recommender(df)
        else:
            recommender = load_advanced_recommender(df, method='word2vec')

        # Auto-generate recommendations (real-time)
        with st.spinner("🔍 Finding recommendations..."):
            try:
                # Get base recommendations
                if selected_ingredients:
                    # Use selected ingredients
                    cleaned_ingredients = ' '.join(selected_ingredients)
                    recommendations = recommender.recommend_by_ingredients(cleaned_ingredients, n_recommendations=20)
                else:
                    # Best choice: first 10 recipes (no computation for speed)
                    recommendations = df.head(10).copy()
                    recommendations['similarity_score'] = 0.85

                if len(recommendations) > 0:
                    # Store original for fallback
                    original_recs = recommendations.copy()

                    # Apply cooking time preference
                    if cooking_time != "Any":
                        filtered = recommendations.copy()
                        if cooking_time == "Quick (<30 min)":
                            filtered = filtered[filtered['minutes'] < 30]
                        elif cooking_time == "Medium (30-60 min)":
                            filtered = filtered[(filtered['minutes'] >= 30) & (filtered['minutes'] <= 60)]
                        elif cooking_time == "Long (>60 min)":
                            filtered = filtered[filtered['minutes'] > 60]
                        if len(filtered) > 0:
                            recommendations = filtered

                    # Apply calorie level preference
                    if calorie_level != "Any":
                        filtered = recommendations.copy()
                        if calorie_level == "Low (<400 cal)":
                            filtered = filtered[filtered['calories'] < 400]
                        elif calorie_level == "Medium (400-600 cal)":
                            filtered = filtered[(filtered['calories'] >= 400) & (filtered['calories'] <= 600)]
                        elif calorie_level == "High (>600 cal)":
                            filtered = filtered[filtered['calories'] > 600]
                        if len(filtered) > 0:
                            recommendations = filtered

                    # Apply dish type preference
                    if dish_type != "Any":
                        filtered = recommendations.copy()
                        dish_keywords = {
                            "Main Course": ["main-dish", "lunch", "dinner"],
                            "Dessert": ["desserts", "sweet"],
                            "Breakfast": ["breakfast", "brunch"],
                            "Side Dish": ["side-dishes"],
                            "Snack": ["snack", "appetizers"],
                            "Salad": ["salads"]
                        }
                        keywords = dish_keywords.get(dish_type, [])
                        if keywords:
                            pattern = '|'.join(keywords)
                            filtered = filtered[filtered['tags_clean'].str.contains(pattern, case=False, na=False)]
                        if len(filtered) > 0:
                            recommendations = filtered

                    # Apply dietary restriction
                    if dietary_restriction != "Any":
                        filtered = recommendations.copy()
                        dietary_keywords = {
                            "Vegetarian": ["vegetarian"],
                            "Vegan": ["vegan"],
                            "Low Fat": ["low-fat", "low-calorie"],
                            "Healthy": ["healthy", "low-sodium"]
                        }
                        keywords = dietary_keywords.get(dietary_restriction, [])
                        if keywords:
                            pattern = '|'.join(keywords)
                            filtered = filtered[filtered['tags_clean'].str.contains(pattern, case=False, na=False)]
                        if len(filtered) > 0:
                            recommendations = filtered

                    # Sort by similarity (highest first)
                    recommendations = recommendations.sort_values('similarity_score', ascending=False)

                    # Limit to top 10
                    recommendations = recommendations.head(10)

                    if len(recommendations) == 0:
                        st.warning("⚠️ No matches found. Showing popular choices...")
                        recommendations = original_recs.sort_values('similarity_score', ascending=False).head(10)

                    # Check if any criteria is applied
                    has_criteria = (
                        dish_type != "Any" or
                        cooking_time != "Any" or
                        dietary_restriction != "Any" or
                        calorie_level != "Any" or
                        len(selected_ingredients) > 0
                    )

                    # Display appropriate message
                    if has_criteria:
                        st.success(f"✨ Found {len(recommendations)} recommendations!")
                    else:
                        st.info("🌟 Our Best Choice")

                    # Display recommendations with expander
                    for idx, (_, recipe) in enumerate(recommendations.iterrows(), 1):
                        # Show match % only when criteria is applied
                        if has_criteria:
                            expander_title = f"**{idx}. {recipe['name']}** - Match: {recipe['similarity_score']:.0%}"
                        else:
                            expander_title = f"**{idx}. {recipe['name']}**"

                        with st.expander(expander_title):
                            # Save to history when expanded
                            if recipe['name'] not in st.session_state.viewed_recipes:
                                st.session_state.viewed_recipes.append(recipe['name'])

                            # Quick info
                            info_cols = st.columns(4)
                            with info_cols[0]:
                                st.metric("Time", f"{recipe['minutes']:.0f} min")
                            with info_cols[1]:
                                st.metric("Calories", f"{recipe['calories']:.0f}")
                            with info_cols[2]:
                                st.metric("Protein", f"{recipe['protein']:.1f}g")
                            with info_cols[3]:
                                st.metric("Steps", f"{recipe.get('n_steps', 'N/A')}")

                            # Tags
                            try:
                                tags = ast.literal_eval(recipe['tags'])
                                st.caption("🏷️ " + ", ".join(tags[:8]))
                            except:
                                pass

                            st.divider()

                            # Cooking Steps
                            st.subheader("📝 Cooking Steps")
                            try:
                                steps = ast.literal_eval(recipe['steps'])
                                for step_idx, step in enumerate(steps, 1):
                                    st.write(f"**{step_idx}.** {step}")
                            except:
                                st.info("No steps available")

                            st.divider()

                            # Ingredients
                            st.subheader("🧺 Ingredients")
                            try:
                                ingredients = ast.literal_eval(recipe['ingredients'])
                                cols = st.columns(2)
                                for i, ing in enumerate(ingredients):
                                    with cols[i % 2]:
                                        st.write(f"• {ing}")
                            except:
                                st.info("No ingredients info")

            except Exception as e:
                st.error(f"Error: {e}")

    # Tab 3: Model Performance
    with tab3:
        st.header("📊 Model Performance Analysis")

        # Select method to evaluate
        st.subheader("⚙️ Select Method to Evaluate")
        selected_method = st.radio(
            "Choose method:",
            ["TF-IDF", "Word2Vec"],
            horizontal=True
        )

        st.divider()

        # Load selected model
        with st.spinner(f"Loading {selected_method} model..."):
            if selected_method == "TF-IDF":
                recommender = load_tfidf_recommender(df)
                embeddings = recommender.tfidf_matrix.toarray()
            else:
                recommender = load_advanced_recommender(df, 'word2vec')
                embeddings = recommender.embeddings

        # Evaluation metrics
        st.subheader("📈 Evaluation Metrics")

        # Sample test recipes
        test_recipes_names = df['name'].sample(min(10, len(df)), random_state=42).tolist()

        with st.spinner("Calculating metrics..."):
            diversities = []
            precisions = []
            recalls = []
            all_recommendations = []

            for recipe_name in test_recipes_names:
                # Get base recipe
                base_recipe_df = df[df['name'] == recipe_name]
                if len(base_recipe_df) == 0:
                    continue
                base_recipe = base_recipe_df.iloc[0]

                # Get recommendations
                recs = recommender.recommend_by_recipe(recipe_name, n_recommendations=10)

                if len(recs) > 0:
                    all_recommendations.append(recs)

                    # Diversity
                    diversity = EvaluationMetrics.calculate_diversity(recs)
                    diversities.append(diversity)

                    # Precision@10
                    precision = EvaluationMetrics.calculate_precision_at_k(base_recipe, recs, k=10)
                    precisions.append(precision)

                    # Recall@10
                    recall = EvaluationMetrics.calculate_recall_at_k(base_recipe, recs, df, k=10)
                    recalls.append(recall)

            # Coverage
            all_recs_df = pd.concat(all_recommendations) if all_recommendations else pd.DataFrame()
            coverage = EvaluationMetrics.calculate_coverage(all_recs_df, df) if len(all_recs_df) > 0 else 0

        # Display metrics in grid
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Diversity", f"{np.mean(diversities):.3f}", help="Variety of tags in recommendations")
            st.metric("Precision@10", f"{np.mean(precisions):.3f}", help="% of relevant items in top 10")

        with col2:
            st.metric("Recall@10", f"{np.mean(recalls):.3f}", help="% of relevant items found")
            st.metric("Coverage", f"{coverage:.2f}%", help="% of catalog recommended")

        # Metrics table
        st.divider()
        st.subheader("Detailed Metrics")

        metrics_data = {
            'Metric': ['Diversity', 'Precision@10', 'Recall@10', 'Coverage'],
            'Value': [
                f"{np.mean(diversities):.4f}",
                f"{np.mean(precisions):.4f}",
                f"{np.mean(recalls):.4f}",
                f"{coverage:.2f}%"
            ],
            'Std Dev': [
                f"{np.std(diversities):.4f}",
                f"{np.std(precisions):.4f}",
                f"{np.std(recalls):.4f}",
                "N/A"
            ]
        }
        st.table(pd.DataFrame(metrics_data))

        st.divider()

        # Embeddings visualization
        st.subheader("🎨 Embeddings Visualization")

        viz_method = st.radio("Visualization Method:", ["t-SNE", "PCA"], horizontal=True)

        if st.button("Generate Visualization", type="primary"):
            with st.spinner(f"Generating {viz_method} visualization..."):
                try:
                    labels = df['name'].tolist()
                    fig = plot_embeddings_tsne(embeddings, labels, method=viz_method, n_samples=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Tab 4: My History
    with tab4:
        st.header("📖 My Recipe History")

        if not st.session_state.viewed_recipes:
            st.info("You haven't viewed any recipes yet. Start exploring in the 'Discover Food' tab!")
        else:
            st.success(f"You have viewed {len(st.session_state.viewed_recipes)} recipes")

            # Display viewed recipes
            st.subheader("📝 Viewed Recipes")
            for idx, recipe_name in enumerate(st.session_state.viewed_recipes, 1):
                st.write(f"{idx}. {recipe_name}")

            st.divider()

            # Personalized recommendations based on viewed history
            st.subheader("🎯 More Recipes You Might Like")
            st.info("Based on your viewing history")

            if st.button("Get Personalized Recommendations", type="primary"):
                with st.spinner("Generating personalized recommendations..."):
                    try:
                        # Load TF-IDF recommender
                        rec = load_tfidf_recommender(df)

                        # Get recommendations based on first viewed recipe
                        base_recipe = st.session_state.viewed_recipes[0]
                        recommendations = rec.recommend_by_recipe(base_recipe, n_recommendations=5)

                        # Filter out already viewed recipes
                        recommendations = recommendations[
                            ~recommendations['name'].isin(st.session_state.viewed_recipes)
                        ]

                        for idx, (_, recipe) in enumerate(recommendations.iterrows(), 1):
                            st.write(f"**{idx}. {recipe['name']}**")
                            st.write(f"   ⏱️ {recipe['minutes']:.0f} min | "
                                   f"🔥 {recipe['calories']:.0f} cal | "
                                   f"💪 {recipe['protein']:.1f}g protein")
                            st.divider()

                    except Exception as e:
                        st.error(f"Error generating personalized recommendations: {e}")

            # Clear history
            st.divider()
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.viewed_recipes = []
                st.success("History cleared!")
                st.rerun()

    # Footer
    st.divider()
    st.caption("Food Recommendation System | Built with Streamlit, scikit-learn, and Gensim")


if __name__ == "__main__":
    main()
