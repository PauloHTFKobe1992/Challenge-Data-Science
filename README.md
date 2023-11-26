# Challenge-Data-Science
Códigos do challenge de DS

!pip install unidecode
from unidecode import unidecode

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt



dados_base = pd.read_csv('/content/items_titles.csv')
dados_comparativos = pd.read_csv('/content/items_titles_test.csv')

'''dados_base = a.head(10)
dados_base'''

'''dados_comparativos = b.head(10)
dados_comparativos'''

def preprocess_text(text):
    text = text.lower()
    text = unidecode(''.join(e for e in text if e.isalnum() or e.isspace()))
    return text

dados_base = dados_base.dropna(subset=['ITE_ITEM_TITLE'])

dados_comparativos = dados_comparativos.dropna(subset=['ITE_ITEM_TITLE'])

texto_a_ser_comparado = dados_comparativos['ITE_ITEM_TITLE'].values
texto_fonte = dados_base['ITE_ITEM_TITLE'].values


print(texto_a_ser_comparado)

print(texto_fonte)

texto_a_ser_comparado = [preprocess_text(text) for text in texto_a_ser_comparado]
texto_fonte = [preprocess_text(text) for text in texto_fonte]


print(texto_fonte)

print(texto_a_ser_comparado)

n = 1

counts = CountVectorizer(analyzer='word', ngram_range=(n, n))

vetores_a_ser_comparado = counts.fit_transform(texto_a_ser_comparado)
vetores_fonte = counts.transform(texto_fonte)

similaridade_cosseno = cosine_similarity(vetores_a_ser_comparado, vetores_fonte)

print(similaridade_cosseno)

df_similaridade = pd.DataFrame(similaridade_cosseno, columns=dados_base['ITE_ITEM_TITLE'], index=dados_comparativos['ITE_ITEM_TITLE'])


df_similaridade['Similaridade Média'] = similaridade_cosseno.mean(axis=1)

print(df_similaridade)


similaridade_media_teste_base = df_similaridade['Similaridade Média'].mean()
similaridade_media_teste_base


df_similaridade = pd.DataFrame(similaridade_cosseno, columns=dados_base['ITE_ITEM_TITLE'], index=dados_comparativos['ITE_ITEM_TITLE'])

# Calcular a média de similaridade para cada item comparativo
df_similaridade['Similaridade Média'] = similaridade_cosseno.mean(axis=1)

# Imprimir DataFrame de similaridade
print(df_similaridade)

# Calcular a média geral de similaridade
similaridade_media_teste_base = df_similaridade['Similaridade Média'].mean()
print("Similaridade Média Geral:", similaridade_media_teste_base)

# Criar um heatmap usando seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(df_similaridade, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5)
plt.title('Mapa de Calor da Similaridade Cosseno entre Textos')
plt.show()

