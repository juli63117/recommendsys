import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import json
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # --- 1. Загрузка и подготовка данных ---
    df = pd.read_csv("bank_feedback_dataset.csv")
    ratings_cols = [f"Q{i}_Score" for i in range(1, 7)]
    user_item_matrix = df[ratings_cols].copy()
    user_ids = user_item_matrix.index.to_list()

    # --- 2. Подготовка данных ---
    user_item_np = user_item_matrix.to_numpy()
    item_medians_np = np.median(user_item_np, axis=0)
    UF = 1 - 1 / (1 + np.abs(user_item_np - item_medians_np[None, :]).mean(axis=0))

    adjusted_scores_np = user_item_np.copy()
    for i in range(adjusted_scores_np.shape[0]):
        for j in range(adjusted_scores_np.shape[1]):
            r = adjusted_scores_np[i, j]
            if r > item_medians_np[j]:
                adjusted_scores_np[i, j] += UF[j]
            elif r < item_medians_np[j]:
                adjusted_scores_np[i, j] -= UF[j]

    user_means_np = user_item_np.mean(axis=1)
    
    # --- 3. Item-based модель (NHSM) ---
    def nhsm_item_similarity(i1, i2):
        users_rated_i1 = ~np.isnan(user_item_np[:, i1])
        users_rated_i2 = ~np.isnan(user_item_np[:, i2])
        common_idx = np.where(users_rated_i1 & users_rated_i2)[0]
        if len(common_idx) == 0:
            return 0.0

        a = adjusted_scores_np[common_idx, i1]
        b = adjusted_scores_np[common_idx, i2]
        prox = 1 - sigmoid(np.abs(a - b))

        item1_r = user_item_np[common_idx, i1]
        item2_r = user_item_np[common_idx, i2]
        mean_users = user_means_np[common_idx]
        sing = sigmoid(np.abs(item1_r - mean_users) * np.abs(item2_r - mean_users))

        sig = (
            (np.abs(item1_r - item_medians_np[i1]) > 1) &
            (np.abs(item2_r - item_medians_np[i2]) > 1)
        ).astype(float)

        pss = prox * sig * sing

        mu1, mu2 = user_item_np[:, i1].mean(), user_item_np[:, i2].mean()
        std1, std2 = user_item_np[:, i1].std(), user_item_np[:, i2].std()
        urp = 1 - sigmoid(np.abs(mu1 - mu2) * np.abs(std1 - std2))

        intersection = len(common_idx)
        union = np.sum(users_rated_i1 | users_rated_i2)
        jaccard = intersection / union if union != 0 else 0.0

        return float(pss.mean()) * urp * jaccard

    item_sim = pd.DataFrame(index=ratings_cols, columns=ratings_cols, dtype=float)
    for i1, i2 in combinations_with_replacement(ratings_cols, 2):
        idx1, idx2 = ratings_cols.index(i1), ratings_cols.index(i2)
        sim = 1.0 if i1 == i2 else nhsm_item_similarity(idx1, idx2)
        item_sim.loc[i1, i2] = sim
        item_sim.loc[i2, i1] = sim

    # Item-based предсказания
    item_based_scores = np.zeros_like(user_item_np)
    for user_idx in range(len(user_item_np)):
        for target_item in range(len(ratings_cols)):
            sim_sum = 0.0
            weighted_sum = 0.0
            for other_item in range(len(ratings_cols)):
                if target_item == other_item:
                    continue
                r = user_item_np[user_idx, other_item]
                if not np.isnan(r):
                    sim = item_sim.iloc[target_item, other_item]
                    weighted_sum += sim * r
                    sim_sum += sim
            item_based_scores[user_idx, target_item] = (
                weighted_sum / sim_sum if sim_sum > 0 else np.nan
            )

    # --- 4. Content-based модель ---
    df['target'] = df[ratings_cols].mean(axis=1)
    cat_cols = ['ServiceType']
    num_cols = ['ServiceCost']
    bool_cols = ['WasStaffPolite', 'IssueResolved', 'UsedMobileBanking']

    # Target Encoding
    te = TargetEncoder()
    df[cat_cols] = te.fit_transform(df[cat_cols], df['target'])

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cat_cols + num_cols + bool_cols])

    # PCA
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # KNN
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(X_pca)
    distances, indices = knn.kneighbors(X_pca)

    # Content-based предсказания
    content_based_scores = np.zeros_like(user_item_np)
    for i, neighbor_idxs in enumerate(indices):
        content_based_scores[i] = df.iloc[neighbor_idxs][ratings_cols].mean().values

    # --- 5. Мета-модель (LightGBM) ---
    te_meta = TargetEncoder()
    df[cat_cols] = te_meta.fit_transform(df[cat_cols], df['target'])
    
    scaler_meta = StandardScaler()
    X_meta = scaler_meta.fit_transform(df[cat_cols + num_cols + bool_cols])
    
    # Обучение мета-модели
    meta_models = []
    for col_idx in range(len(ratings_cols)):
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        X = np.column_stack([
            user_item_np[:, col_idx],
            df[ratings_cols].mean(axis=1),
            X_meta
        ])
        y = df[ratings_cols[col_idx]]
        valid_idx = ~np.isnan(y)
        if sum(valid_idx) > 10:
            model.fit(X[valid_idx], y[valid_idx])
        meta_models.append(model)
    
    # Предсказания гибридной модели
    hybrid_scores = np.zeros_like(user_item_np)
    for col_idx, model in enumerate(meta_models):
        if model is not None:
            X = np.column_stack([
                user_item_np[:, col_idx],
                df[ratings_cols].mean(axis=1),
                X_meta
            ])
            hybrid_scores[:, col_idx] = model.predict(X)
    
    # Заполнение пропусков
    hybrid_scores = np.where(np.isnan(hybrid_scores), content_based_scores, hybrid_scores)
    hybrid_scores = np.where(np.isnan(hybrid_scores), item_based_scores, hybrid_scores)

    # --- 6. Генерация рекомендаций ---
    question_labels = {
        "Q1_Score": "Доступность услуг",
        "Q2_Score": "Качество консультации", 
        "Q3_Score": "Уровень цифрового сервиса (сайт, приложение)",
        "Q4_Score": "Время ожидания обслуживания",
        "Q5_Score": "Прозрачность тарифов и условий",
        "Q6_Score": "Уровень доверия к банку (надежность, безопасность)"
    }

    # --- 6. Генерация рекомендаций ---
    question_labels = {
        "Q1_Score": "Доступность услуг",
        "Q2_Score": "Качество консультации", 
        "Q3_Score": "Уровень цифрового сервиса (сайт, приложение)",
        "Q4_Score": "Время ожидания обслуживания",
        "Q5_Score": "Прозрачность тарифов и условий",
        "Q6_Score": "Уровень доверия к банку (надежность, безопасность)"
    }

    min_score, max_score = 1, 10
    hybrid_scores_scaled = np.clip(hybrid_scores, min_score, max_score)
    threshold = 6  # Пороговое значение

    recommendations_json = []

    for user_idx in range(len(hybrid_scores_scaled)):
        user_recs = []
        for col_idx, col in enumerate(ratings_cols):
            score = hybrid_scores_scaled[user_idx, col_idx]
            if score < threshold:
                user_recs.append(f"Нужно улучшить: {question_labels[col]}")
        
        # Только если есть рекомендации
        if user_recs:
            recommendations_json.append({
                "user_id": int(user_ids[user_idx]),  # или user_idx + 1, если ID нет
                "recommendations": user_recs
            })

    # Пример вывода в консоль
    print("\n=== Первые 3 рекомендации ===")
    for user in recommendations_json[:3]:
        print(f"\nПользователь {user['user_id']}:")
        for msg in user["recommendations"]:
            print(f"- {msg}")

    # Сохранение в JSON
    output_path = r"C:\webPJ-main\sys\server\recommender\output\recommendations.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recommendations_json, f, ensure_ascii=False, indent=2)

    print(f"\nВсе рекомендации сохранены в: {output_path}")


if __name__ == "__main__":
    main()