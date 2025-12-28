import ast
import re
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def parse_nutrition(nutrition_str):
    """
    Parse nutrition string to extract individual values.

    Args:
        nutrition_str: String in format "[calories, fat, sugar, sodium, protein, saturated_fat, carbs]"

    Returns:
        dict: Dictionary with nutrition values
    """
    try:
        nutrition_list = ast.literal_eval(nutrition_str)
        return {
            'calories': float(nutrition_list[0]),
            'fat': float(nutrition_list[1]),
            'sugar': float(nutrition_list[2]),
            'sodium': float(nutrition_list[3]),
            'protein': float(nutrition_list[4]),
            'saturated_fat': float(nutrition_list[5]),
            'carbs': float(nutrition_list[6])
        }
    except:
        return {
            'calories': 0, 'fat': 0, 'sugar': 0,
            'sodium': 0, 'protein': 0, 'saturated_fat': 0, 'carbs': 0
        }


def clean_ingredients(ingredients_str):
    """
    Clean and process ingredients string.

    Args:
        ingredients_str: String representation of ingredients list

    Returns:
        str: Cleaned ingredients as space-separated string
    """
    try:
        ingredients_list = ast.literal_eval(ingredients_str)
        cleaned = [re.sub(r'[^a-zA-Z\s]', '', ing.lower().strip()) for ing in ingredients_list]
        return ' '.join(cleaned)
    except:
        return ''


def clean_tags(tags_str):
    """
    Clean and process tags string.

    Args:
        tags_str: String representation of tags list

    Returns:
        str: Cleaned tags as space-separated string
    """
    try:
        tags_list = ast.literal_eval(tags_str)
        cleaned = [tag.lower().strip() for tag in tags_list]
        return ' '.join(cleaned)
    except:
        return ''


def explain_recommendation(recipe, base_recipe, similarity_score):
    """
    Generate explanation for why a recipe was recommended.

    Args:
        recipe: Recommended recipe row
        base_recipe: Base recipe row
        similarity_score: Similarity score between recipes

    Returns:
        str: Explanation text
    """
    explanations = []

    # Compare ingredients
    try:
        base_ingredients = set(ast.literal_eval(base_recipe['ingredients']))
        rec_ingredients = set(ast.literal_eval(recipe['ingredients']))
        common_ingredients = base_ingredients.intersection(rec_ingredients)
        if len(common_ingredients) > 0:
            explanations.append(f"{len(common_ingredients)} shared ingredients")
    except:
        pass

    # Compare tags
    try:
        base_tags = set(ast.literal_eval(base_recipe['tags']))
        rec_tags = set(ast.literal_eval(recipe['tags']))
        common_tags = base_tags.intersection(rec_tags)
        if len(common_tags) > 0:
            explanations.append(f"{len(common_tags)} matching tags")
    except:
        pass

    # Compare cooking time
    time_diff = abs(recipe['minutes'] - base_recipe['minutes'])
    if time_diff < 10:
        explanations.append("similar cooking time")

    # Compare calories
    cal_diff = abs(recipe['calories'] - base_recipe['calories'])
    if cal_diff < 100:
        explanations.append("similar calorie count")

    if not explanations:
        explanations.append(f"{similarity_score:.1%} similarity")

    return ", ".join(explanations[:3])


def plot_embeddings_tsne(embeddings, labels, method='t-SNE', n_samples=1000):
    """
    Create t-SNE or PCA visualization of embeddings.

    Args:
        embeddings: Embedding vectors (2D array)
        labels: Labels for each embedding
        method: 't-SNE' or 'PCA'
        n_samples: Number of samples to visualize

    Returns:
        plotly figure
    """
    # Sample if too many points
    if len(embeddings) > n_samples:
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sample = embeddings[indices]
        labels_sample = [labels[i] for i in indices]
    else:
        embeddings_sample = embeddings
        labels_sample = labels

    # Dimensionality reduction
    if method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
    else:
        reducer = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer.fit_transform(embeddings_sample)

    # Create DataFrame
    df_viz = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'recipe': labels_sample
    })

    # Create plot
    fig = px.scatter(
        df_viz,
        x='x',
        y='y',
        hover_data=['recipe'],
        title=f'Recipe Similarity Map ({method})',
        labels={
            'x': 'Dimension 1 (Recipe Features)',
            'y': 'Dimension 2 (Recipe Features)',
            'recipe': 'Recipe Name'
        }
    )

    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.update_layout(
        height=600,
        xaxis_title='Dimension 1 (Recipe Features)',
        yaxis_title='Dimension 2 (Recipe Features)',
        font=dict(size=12)
    )

    return fig


def filter_by_context(df, meal_context=None, time_context=None, health_context=None):
    """
    Filter recipes by context.

    Args:
        df: Recipe DataFrame
        meal_context: 'breakfast', 'lunch', 'dinner', 'snack'
        time_context: 'quick' (<30), 'medium' (30-60), 'long' (>60)
        health_context: 'low_calorie', 'high_protein', 'balanced'

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Meal context
    if meal_context:
        meal_keywords = {
            'breakfast': ['breakfast', 'morning', 'brunch'],
            'lunch': ['lunch', 'midday'],
            'dinner': ['dinner', 'supper', 'main-dish'],
            'snack': ['snack', 'appetizer', 'finger-food']
        }
        keywords = meal_keywords.get(meal_context, [])
        if keywords:
            filtered_df = filtered_df[filtered_df['tags_clean'].str.contains('|'.join(keywords), case=False, na=False)]

    # Time context
    if time_context == 'quick':
        filtered_df = filtered_df[filtered_df['minutes'] < 30]
    elif time_context == 'medium':
        filtered_df = filtered_df[(filtered_df['minutes'] >= 30) & (filtered_df['minutes'] <= 60)]
    elif time_context == 'long':
        filtered_df = filtered_df[filtered_df['minutes'] > 60]

    # Health context
    if health_context == 'low_calorie':
        filtered_df = filtered_df[filtered_df['calories'] < 400]
    elif health_context == 'high_protein':
        filtered_df = filtered_df[filtered_df['protein'] > 20]
    elif health_context == 'balanced':
        filtered_df = filtered_df[
            (filtered_df['calories'] >= 300) &
            (filtered_df['calories'] <= 600) &
            (filtered_df['protein'] >= 15)
        ]

    return filtered_df


def create_visualizations(df):
    """
    Create all required visualizations.

    Args:
        df: Recipe DataFrame with nutrition columns

    Returns:
        dict: Dictionary of plotly figures
    """
    figs = {}

    # Cooking time distribution
    figs['cooking_time'] = px.histogram(
        df[df['minutes'] <= 120],
        x='minutes',
        nbins=50,
        title='Cooking Time Distribution (≤120 minutes)',
        labels={'minutes': 'Cooking Time (minutes)', 'count': 'Number of Recipes'}
    )
    figs['cooking_time'].update_layout(showlegend=False)

    # Top 10 common tags
    all_tags = []
    for tags_str in df['tags']:
        try:
            tags_list = ast.literal_eval(tags_str)
            all_tags.extend(tags_list)
        except:
            pass

    tag_counts = pd.Series(all_tags).value_counts().head(10)
    figs['top_tags'] = px.bar(
        x=tag_counts.index,
        y=tag_counts.values,
        title='Top 10 Most Common Recipe Tags',
        labels={'x': 'Tag', 'y': 'Frequency'}
    )

    # Nutrition scatter: calories vs protein
    figs['nutrition_scatter'] = px.scatter(
        df[df['calories'] <= 2000],
        x='calories',
        y='protein',
        hover_data=['name'],
        title='Nutrition Profile: Calories vs Protein',
        labels={'calories': 'Calories', 'protein': 'Protein (g)'},
        opacity=0.5
    )

    # Correlation heatmap
    nutrition_cols = ['calories', 'fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbs']
    corr_matrix = df[nutrition_cols].corr()
    figs['correlation'] = px.imshow(
        corr_matrix,
        title='Nutrition Correlation Heatmap',
        labels={'color': 'Correlation'},
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )

    return figs
