"""
unsupervised_learning.py
------------------------
Modelos de aprendizaje NO SUPERVISADO para el sistema de transporte masivo.

Técnicas implementadas:
1. K-Means Clustering - Agrupar viajes por comportamiento similar
2. DBSCAN - Detectar viajes anómalos (outliers)
3. PCA - Reducción de dimensionalidad y visualización
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Configuración
np.random.seed(42)

# ==================== 1. CARGAR DATOS ====================
print("="*60)
print("1. CARGANDO DATASET")
print("="*60)

df = pd.read_csv('dataset_transporte_unsupervised.csv')
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas disponibles: {list(df.columns)}")

# ==================== 2. PREPROCESAMIENTO ====================
print("\n" + "="*60)
print("2. PREPROCESAMIENTO DE DATOS")
print("="*60)

# Codificar variables categóricas (estaciones)
le_origen = LabelEncoder()
le_destino = LabelEncoder()

df['origen_enc'] = le_origen.fit_transform(df['estacion_origen'])
df['destino_enc'] = le_destino.fit_transform(df['estacion_destino'])

# Seleccionar features numéricas para clustering
features_numericas = [
    'hora_del_dia', 'dia_semana', 'num_transbordos', 
    'pasajeros_en_sistema', 'tiempo_estimado_min', 
    'origen_enc', 'destino_enc'
]

X = df[features_numericas].values

# Escalar datos (importante para distancias en clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features para clustering: {features_numericas}")
print(f"Shape de matriz de features: {X_scaled.shape}")

# ==================== 3. K-MEANS CLUSTERING ====================
print("\n" + "="*60)
print("3. K-MEANS CLUSTERING - Agrupación de viajes similares")
print("="*60)

# Encontrar número óptimo de clusters usando el método del codo
inercia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Mostrar resultados
print("\nMétodo del codo - Inercia por número de clusters:")
for k, iner in zip(K_range, inercia):
    print(f"  K={k}: Inercia={iner:.0f}")

# Elegir K=4 (valor razonable basado en tipos de viaje)
k_optimo = 4
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)

print(f"\n✅ K-means aplicado con K={k_optimo}")
print(f"Silhouette Score (K={k_optimo}): {silhouette_scores[k_optimo-2]:.4f}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, df['cluster_kmeans']):.4f}")

# Interpretar clusters
print("\n📊 Interpretación de clusters (K-means):")
for i in range(k_optimo):
    cluster_data = df[df['cluster_kmeans'] == i]
    print(f"\n  Cluster {i}: {len(cluster_data)} viajes ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"    - Hora promedio: {cluster_data['hora_del_dia'].mean():.1f}h")
    print(f"    - Pasajeros promedio: {cluster_data['pasajeros_en_sistema'].mean():.0f}")
    print(f"    - Transbordos promedio: {cluster_data['num_transbordos'].mean():.2f}")

# ==================== 4. DBSCAN - DETECCIÓN DE ANOMALÍAS ====================
print("\n" + "="*60)
print("4. DBSCAN - Detección de viajes anómalos (outliers)")
print("="*60)

# Probar diferentes parámetros de epsilon
eps_values = [0.5, 0.8, 1.0, 1.2, 1.5]
min_samples = 5

print("Probando diferentes valores de eps:")
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = sum(labels == -1)
    print(f"  eps={eps}: {n_clusters} clusters, {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

# Usar eps=1.0 como valor razonable
eps_final = 1.0
dbscan = DBSCAN(eps=eps_final, min_samples=5)
df['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
outliers = df[df['cluster_dbscan'] == -1]
n_outliers = len(outliers)

print(f"\n✅ DBSCAN aplicado con eps={eps_final}, min_samples=5")
print(f"  Viajes normales: {len(df) - n_outliers}")
print(f"  Viajes anómalos (outliers): {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

if n_outliers > 0:
    print("\n📊 Ejemplos de viajes anómalos:")
    print(outliers[['estacion_origen', 'estacion_destino', 'hora_del_dia', 
                    'pasajeros_en_sistema', 'num_transbordos']].head(10))

# ==================== 5. PCA - VISUALIZACIÓN ====================
print("\n" + "="*60)
print("5. PCA - Reducción de dimensionalidad para visualización")
print("="*60)

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Varianza explicada por componente 1: {pca.explained_variance_ratio_[0]:.4f} ({pca.explained_variance_ratio_[0]*100:.2f}%)")
print(f"Varianza explicada por componente 2: {pca.explained_variance_ratio_[1]:.4f} ({pca.explained_variance_ratio_[1]*100:.2f}%)")
print(f"Varianza total explicada: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")

# Interpretación de componentes
print("\nContribución de variables a los componentes principales:")
componentes_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=features_numericas
)
print(componentes_df.round(4))

# ==================== 6. VISUALIZACIONES ====================
print("\n" + "="*60)
print("6. GENERANDO GRÁFICAS")
print("="*60)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Gráfica 1: Método del codo (K-means)
axes[0, 0].plot(K_range, inercia, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axvline(x=k_optimo, color='r', linestyle='--', label=f'K elegido = {k_optimo}')
axes[0, 0].set_xlabel('Número de Clusters (K)')
axes[0, 0].set_ylabel('Inercia')
axes[0, 0].set_title('Método del Codo - K-Means')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfica 2: Silhouette Score
axes[0, 1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].axvline(x=k_optimo, color='r', linestyle='--', label=f'K elegido = {k_optimo}')
axes[0, 1].set_xlabel('Número de Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score por Número de Clusters')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gráfica 3: Visualización de clusters K-means con PCA
scatter1 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                              c=df['cluster_kmeans'], cmap='viridis', 
                              alpha=0.6, s=30)
axes[1, 0].set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1, 0].set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1, 0].set_title(f'Clusters K-means (K={k_optimo}) visualizados con PCA')
plt.colorbar(scatter1, ax=axes[1, 0], label='Cluster')

# Gráfica 4: Visualización de outliers DBSCAN
outlier_colors = ['red' if label == -1 else 'blue' for label in df['cluster_dbscan']]
axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=outlier_colors, alpha=0.6, s=30)
axes[1, 1].set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1, 1].set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1, 1].set_title(f'Detección de Anomalías - DBSCAN (Outliers en rojo)')
axes[1, 1].legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Normal', markersize=8),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Outlier', markersize=8)
])

plt.tight_layout()
plt.savefig('resultados_unsupervised.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 7. GUARDAR RESULTADOS ====================
print("\n" + "="*60)
print("7. GUARDANDO RESULTADOS")
print("="*60)

df.to_csv('dataset_con_clusters.csv', index=False)
print("✅ Dataset con clusters guardado: dataset_con_clusters.csv")
print("✅ Gráfica guardada: resultados_unsupervised.png")

# ==================== 8. RESUMEN FINAL ====================
print("\n" + "="*60)
print("8. RESUMEN DE RESULTADOS")
print("="*60)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  TÉCNICA                │  HALLAZGO                                │
├─────────────────────────────────────────────────────────────────────┤
│  K-Means Clustering     │  Se identificaron {k_optimo} tipos de viajes:              │
│                         │    • Cluster 0: Viajes con características específicas     │
│                         │    • Cluster 1: Patrón diferente        │
│                         │    • Cluster 2: Otro comportamiento     │
│                         │    • Cluster 3: Comportamiento atípico  │
├─────────────────────────────────────────────────────────────────────┤
│  DBSCAN                 │  Se detectaron {n_outliers} viajes anómalos              │
│  (Detección anomalías)  │  ({n_outliers/len(df)*100:.1f}% del total)                    │
├─────────────────────────────────────────────────────────────────────┤
│  PCA                    │  Las 2 primeras componentes explican     │
│  (Reducción dim)        │  el {sum(pca.explained_variance_ratio_)*100:.1f}% de la variabilidad   │
└─────────────────────────────────────────────────────────────────────┘
""")