import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import umap

cat_folder = 'imagem'  
test_folder = 'imagem_test'  
k = 5  
contamination = 0.1

def load_images_from_folder(folder_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    images, file_names = [], []
    for file in tqdm(os.listdir(folder_path), desc=f"Carregando {folder_path}"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert('RGB')
                images.append(transform(img))
                file_names.append(img_path)
            except:
                print(f"Erro ao carregar: {file}")
    return images, file_names

# extrair embeddings
def extract_embeddings(images):
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for img in tqdm(images, desc="Extraindo embeddings"):
            img = img.unsqueeze(0)
            emb = model(img).squeeze().numpy()
            embeddings.append(emb)
    
    return np.array(embeddings)

# imagens de treinamento (apenas gatos)
cat_images, cat_file_names = load_images_from_folder(cat_folder)
cat_embeddings = extract_embeddings(cat_images)

def get_race_from_filename(filename):
    for r in ['cornish_rex', 'devon_rex', 'main_coon', 'mixed', 'sphynx']:
        if r in filename.lower():
            return r
    return 'unknown'

cat_races = [get_race_from_filename(fn) for fn in cat_file_names]



# standard dos embeddings
scaler = StandardScaler()
cat_embeddings_scaled = scaler.fit_transform(cat_embeddings)

# one-Class 
one_class_svm = OneClassSVM(
    kernel='rbf', 
    gamma='scale', 
    nu=contamination 
)
one_class_svm.fit(cat_embeddings_scaled)
# clusterização com KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
cat_labels = kmeans.fit_predict(cat_embeddings_scaled)

# visualizar clusters
def plot_clusters(file_names, labels, title="Clusters", max_imgs=8):
    for cluster in sorted(set(labels)):
        plt.figure(figsize=(12, 3))
        plt.suptitle(f'{title} - Cluster {cluster}')
        imgs = [fn for fn, lbl in zip(file_names, labels) if lbl == cluster][:max_imgs]
        
        for i, img_path in enumerate(imgs):
            img = Image.open(img_path).resize((100, 100))
            plt.subplot(1, max_imgs, i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(img_path), fontsize=8)
        
        plt.tight_layout()
        plt.show()


plot_clusters(cat_file_names, cat_labels, "Clusters de Gatos")

if os.path.exists(test_folder):
    test_images, test_file_names = load_images_from_folder(test_folder)
    test_embeddings = extract_embeddings(test_images)
    test_embeddings_scaled = scaler.transform(test_embeddings)
    
    # 1 = normal (gato), -1 = outlier (não-gato/cachorro)
    predictions = one_class_svm.predict(test_embeddings_scaled)
    
    cat_detected = [(fn, pred) for fn, pred in zip(test_file_names, predictions) if pred == 1]
    dog_detected = [(fn, pred) for fn, pred in zip(test_file_names, predictions) if pred == -1]
    
    print(f"Imagens classificadas como GATOS: {len(cat_detected)}")
    print(f"Imagens classificadas como DESCONHECIDAS/CACHORROS: {len(dog_detected)}")
    
    def plot_classification_results(detected_list, title, max_imgs=8):
        if detected_list:
            plt.figure(figsize=(12, 3))
            plt.suptitle(f'{title} ({len(detected_list)} imagens)')
            
            for i, (img_path, _) in enumerate(detected_list[:max_imgs]):
                img = Image.open(img_path).resize((100, 100))
                plt.subplot(1, max_imgs, i+1)
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.basename(img_path), fontsize=8)
            
            plt.tight_layout()
            plt.show()
    
    plot_classification_results(cat_detected, "Classificadas como GATOS")
    plot_classification_results(dog_detected, "Classificadas como DESCONHECIDAS/CACHORROS")
    
    decision_scores = one_class_svm.decision_function(test_embeddings_scaled)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(decision_scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Limiar de decisão')
    plt.xlabel('Pontuação de Decisão')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Pontuações de Decisão')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    colors = ['green' if pred == 1 else 'red' for pred in predictions]
    plt.scatter(range(len(decision_scores)), decision_scores, c=colors, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', label='Limiar de decisão')
    plt.xlabel('Índice da Imagem')
    plt.ylabel('Pontuação de Decisão')
    plt.title('Pontuações por Imagem')
    plt.legend(['Limiar', 'Gatos', 'Desconhecidas'])
    
    plt.tight_layout()
    plt.show()
    
    confidence_results = list(zip(test_file_names, predictions, decision_scores))
    confidence_results.sort(key=lambda x: x[2], reverse=True)
    
    print("\n GATOS ")
    for fn, pred, score in confidence_results[:5]:
        print(f"{os.path.basename(fn)}: {score:.3f} ({'GATO' if pred == 1 else 'DESCONHECIDA'})")
    
    print("\n DESCONHECIDAS")
    for fn, pred, score in confidence_results[-5:]:
        print(f"{os.path.basename(fn)}: {score:.3f} ({'GATO' if pred == 1 else 'DESCONHECIDA'})")

else:
    print(f"\nPasta de teste '{test_folder}' não encontrada.")
    print("Crie uma pasta com imagens mistas (gatos + cachorros) para testar o classificador!")

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(cat_embeddings_scaled)-1))
cat_embeddings_2d = tsne.fit_transform(cat_embeddings_scaled)

import seaborn as sns

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=cat_embeddings_2d[:, 0],
    y=cat_embeddings_2d[:, 1],
    hue=cat_races,
    palette='tab10',
    s=60,
    alpha=0.8
)
plt.title("t-SNE dos Gatos Agrupados por Raça")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend(title='Raça', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


umap_model = umap.UMAP(n_components=2, random_state=42)
cat_embeddings_umap = umap_model.fit_transform(cat_embeddings_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=cat_embeddings_umap[:, 0],
    y=cat_embeddings_umap[:, 1],
    hue=cat_races,
    palette='tab10',
    s=60,
    alpha=0.8
)
plt.title("UMAP dos Gatos Agrupados por Raça")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend(title='Raça', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
