
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def carregar_dados(arquivo_excel, sheet_name, target):
    df = pd.read_excel(arquivo_excel, sheet_name=sheet_name)
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def preencher_com_media_coluna(col):
    col_mean = col.mean()
    if pd.isna(col_mean):
        col_mean = 0
    col = col.fillna(col_mean)
    return col

def processar_planilha(arquivo_entrada, arquivo_saida, sheets_names):
    xlsx = pd.ExcelFile(arquivo_entrada)
    nomes_planilhas = xlsx.sheet_names
    dfs_preprocessados = {}
    for nome_planilha, sheets_name in zip(nomes_planilhas, sheets_names):
        df_planilha = pd.read_excel(xlsx, sheet_name=nome_planilha)
        df_preprocessado = df_planilha.apply(preencher_com_media_coluna, axis=0)
        dfs_preprocessados[sheets_name] = df_preprocessado
    xlsx.close()
    with pd.ExcelWriter(arquivo_saida, engine='xlsxwriter') as writer:
        for sheets_name, df_preprocessado in dfs_preprocessados.items():
            df_preprocessado.to_excel(writer, sheet_name=sheets_name, index=False)

def treinar_modelo(X_treino, y_treino, parametros_modelo=None):
    if parametros_modelo is None:
        parametros_modelo = {
            "n_estimators": 100,
            "min_samples_split": 5,
            "min_samples_leaf": 3,
            "random_state": 42
        }
    modelo = RandomForestClassifier(**parametros_modelo)
    modelo.fit(X_treino, y_treino)
    return modelo

def ajustar_modelo(X_treino, y_treino, parametros_grid=None):
    if parametros_grid is None:
        parametros_grid = {
            "n_estimators": [100, 250, 500],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, 'sqrt', 'log2']
        }
    modelo = RandomForestClassifier()
    grid = GridSearchCV(modelo, parametros_grid, scoring='f1_micro', cv=5)
    grid.fit(X_treino, y_treino)
    print("Melhores parâmetros encontrados:")
    print(grid.best_params_)
    return grid.best_estimator_

def avaliar_modelo(modelo, X, y, cv=5):
    scores = cross_val_score(modelo, X, y, cv=cv, scoring="f1_micro")
    print(f"Acurácia média em {cv}-fold cross-validation: {scores.mean()}")

def comparar_resultados(modelo, sheet_name, X_teste, y_teste, y_pred):
    print(f"Resultados do Modelo para {sheet_name}:")
    print("Classification Report:")
    print(classification_report(y_teste, y_pred))
    print(f"Acurácia: {accuracy_score(y_teste, y_pred)}")
    print("\n")

def visualizar_importancias(modelo, feature_names, nome_arquivo):
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]
    plt.figure(figsize=(8, 10))
    sns.barplot(y=feature_names[indices], x=importancias[indices], orient='h')
    plt.xlabel("Importância Relativa")
    plt.ylabel("Atributos")
    plt.title("Importância dos Atributos no Random Forest")
    plt.savefig(nome_arquivo, format='png')

def main():
    arquivo_excel_preprocessado = 'DesafioML_Dataset/preprocess_data/cb12_Preprocessado.xlsx'
    arquivo_excel_parameters = 'DesafioML_Dataset/input_data/dataset_parameters.xlsx'
    sheets_names = ['P_Aceleration', 'P_Frenagem', 'P_Manobrabilidade', 'P_Suspensao', 'P_Tracao', 'P_Velocidade_Final']
    target_names = ['P_Aceleration', 'P_Frenagem', 'P_Manobrabilidade', 'P_Suspensao', 'P_Tracao', 'P_Velocidade_Final']

    # Preprocessamento dos Dados
    processar_planilha(arquivo_excel_parameters, arquivo_excel_preprocessado, sheets_names)

    # Ajustando modelo utilizando GridSearchCV
    for sheet_name, target in zip(sheets_names, target_names):
        X_treino, y_treino, X_teste, y_teste = carregar_dados(arquivo_excel_preprocessado, sheet_name, target)
        X_treino, X_teste, y_treino, y_teste = train_test_split(X_treino, y_treino, test_size=0.3, random_state=42)
        modelo_otimizado = ajustar_modelo(X_treino, y_treino)

        # Visualização e outros processamentos
        y_pred = modelo_otimizado.predict(X_teste)
        comparar_resultados(modelo_otimizado, sheet_name, X_teste, y_teste, y_pred)
        visualizar_importancias(modelo_otimizado, X_treino.columns, f'{sheet_name}_importancia_atributos.png')

        # Aplicar ao modelo nos dados novos
        novo_dataset = 'DesafioML_Dataset/preprocess_data/cb12_Preprocessado.xlsx'
        X_novo, _, y_pred_novo = carregar_dados(novo_dataset, sheet_name, target)
        y_pred_novo = modelo_otimizado.predict(X_novo)
        print(f'Previsões no novo dataset para {sheet_name}: {y_pred_novo}')

if __name__ == "__main__":
    main()
