import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample data (User ratings for movies)
data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Movie1': [5, 4, 0, 3, 2],
    'Movie2': [3, 5, 4, 0, 1],
    'Movie3': [1, 0, 5, 2, 3],
    'Movie4': [4, 2, 3, 1, 5],
    'Movie5': [2, 3, 1, 5, 4]
}

df = pd.DataFrame(data)

# Function to get movie recommendations for a user
def get_movie_recommendations(user_ratings, user_id, df):
    # Calculate cosine similarity between users
    similarity_matrix = cosine_similarity(df.drop('User', axis=1))

    # Find the user index
    user_index = df[df['User'] == user_id].index[0]

    # Get the similarity scores for the user
    user_similarity = similarity_matrix[user_index]

    # Find users similar to the target user
    similar_users = list(enumerate(user_similarity))

    # Sort similar users based on similarity scores
    sorted_similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)

    # Initialize a dictionary to store movie recommendations
    recommendations = {}

    # Iterate over similar users and their ratings
    for user, similarity_score in sorted_similar_users[1:]:  # Exclude the target user
        for i, rating in enumerate(df.iloc[user, 1:]):
            if rating > 0 and df.iloc[user_index, i + 1] == 0:  # Check if the target user has not rated the movie
                if i + 1 not in recommendations:
                    recommendations[i + 1] = similarity_score * rating
                else:
                    recommendations[i + 1] += similarity_score * rating

    # Sort recommendations based on the total weighted ratings
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # Return the top N recommendations
    top_recommendations = [movie for movie, score in sorted_recommendations[:3]]  # You can change the number of recommendations
    return top_recommendations

# Example usage
user_id = 'User1'
user_ratings = df[df['User'] == user_id].drop('User', axis=1).values.flatten()

recommendations = get_movie_recommendations(user_ratings, user_id, df)
print(f"Recommendations for {user_id}: {recommendations}")
