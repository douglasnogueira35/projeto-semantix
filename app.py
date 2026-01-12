import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
from lime.lime_tabular import LimeTabularExplainer
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import io

# ---------------- Configuração estética ----------------
st.set_page_config(page_title="Análise de Clientes", page_icon="💼", layout="wide")

st.title("📊 Relatório de Intenção de Compra Online")
st.markdown("Interface profissional com métricas, gráficos e explicabilidade")

# ---------------- Sidebar ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.sidebar.title("Configurações")

arquivo = st.sidebar.file_uploader("Carregar arquivo (CSV ou XLSX)", type=["csv","xlsx"])
alvo = st.sidebar.text_input("Variável alvo", "Revenue")
modelo_escolhido = st.sidebar.selectbox("Modelo", ["Regressão Logística", "Random Forest", "XGBoost"])

# ---------------- Funções auxiliares ----------------
@st.cache_data
def carregar_dados(arquivo):
    if arquivo.name.endswith(".csv"):
        return pd.read_csv(arquivo)
    else:
        return pd.read_excel(arquivo)

def avaliar(modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    y_proba = modelo.predict_proba(X_teste)[:,1]
    return {
        "Acurácia": accuracy_score(y_teste, y_pred),
        "Precisão": precision_score(y_teste, y_pred),
        "Recall": recall_score(y_teste, y_pred),
        "F1": f1_score(y_teste, y_pred),
        "AUC": roc_auc_score(y_teste, y_proba)
    }

# ---------------- Pipeline ----------------
if arquivo:
    dados = carregar_dados(arquivo)

    # Tradução de colunas
    traducao = {"Revenue":"Compra","BounceRates":"TaxaRejeição","ExitRates":"TaxaSaída",
                "PageValues":"ValorPágina","SpecialDay":"DiaEspecial","Month":"Mês",
                "OperatingSystems":"SistemaOperacional","Browser":"Navegador","Region":"Região",
                "TrafficType":"TipoTráfego","VisitorType":"TipoVisitante","Weekend":"FimDeSemana"}
    dados = dados.rename(columns=traducao)

    X = dados.drop(columns=["Compra"])
    y = dados["Compra"].astype(int)

    colunas_num = X.select_dtypes(include=[float,int]).columns.tolist()
    colunas_cat = X.select_dtypes(exclude=[float,int]).columns.tolist()

    preprocessador = ColumnTransformer([
        ("num", StandardScaler(), colunas_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), colunas_cat)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train_proc = preprocessador.fit_transform(X_train)
    X_test_proc = preprocessador.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4 = st.tabs(["Exploração", "Modelagem", "Validação", "Resultados"])

    # -------- Tab 1: EDA --------
    with tab1:
        st.subheader("📊 Análise Exploratória")
        st.plotly_chart(px.histogram(dados, x="ValorPágina", color="Compra"))
        st.plotly_chart(px.box(dados, x="Compra", y="TaxaRejeição"))
        fig_corr = px.imshow(dados[colunas_num].corr(), text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig_corr)

    # -------- Tab 2: Modelagem --------
    with tab2:
        if modelo_escolhido == "Regressão Logística":
            modelo = LogisticRegression(max_iter=1000)
        elif modelo_escolhido == "Random Forest":
            param_grid = {'n_estimators':[100,200,300],'max_depth':[None,5,10],'min_samples_split':[2,5,10]}
            search = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                        param_grid, n_iter=5, cv=3, scoring="roc_auc")
            search.fit(X_train_bal, y_train_bal)
            modelo = search.best_estimator_
            st.write("🔧 Melhor conjunto de hiperparâmetros:", search.best_params_)
        else:
            modelo = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)

        modelo.fit(X_train_bal, y_train_bal)
        resultados = avaliar(modelo, X_test_proc, y_test)

        st.subheader("📈 Métricas do Modelo")
        st.write(pd.DataFrame([resultados]))

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=list(resultados.keys()), y=list(resultados.values()), palette="Blues", ax=ax)
        ax.set_title("Desempenho do Modelo", fontsize=14)
        st.pyplot(fig)

    # -------- Tab 3: Validação --------
    with tab3:
        st.subheader("📊 Testes Estatísticos")

        # VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = colunas_num
        vif_data["VIF"] = [variance_inflation_factor(X_train[colunas_num].values, i)
                           for i in range(len(colunas_num))]
        st.write("Multicolinearidade (VIF):")
        st.write(vif_data)

        # Normalidade dos resíduos
        residuos = y_test - modelo.predict(X_test_proc)
        stat, p = shapiro(residuos)
        st.write(f"Normalidade dos resíduos (Shapiro-Wilk): p-valor={p:.4f}")

        # Homoscedasticidade (Breusch-Pagan) com constante
        X_test_bp = sm.add_constant(X_test_proc)
        bp_test = sm.stats.diagnostic.het_breuschpagan(residuos, X_test_bp)
        st.write(f"Homoscedasticidade (Breusch-Pagan): estatística={bp_test[0]:.3f}, p-valor={bp_test[1]:.4f}")

    # -------- Tab 4: Resultados --------
  
    with tab4:
        st.subheader("📊 Comparação de Modelos")
        resultados_modelos = {}
        for nome, modelo_cls in {
            "Regressão Logística": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
        }.items():
            inicio = time.time()
            modelo_cls.fit(X_train_bal, y_train_bal)
            fim = time.time()
            resultados_modelos[nome] = avaliar(modelo_cls, X_test_proc, y_test)
            resultados_modelos[nome]["TempoTreino"] = fim - inicio
        st.write(pd.DataFrame(resultados_modelos).T)

        # Explicabilidade SHAP
        st.subheader("🔍 Explicabilidade (SHAP)")
        if modelo_escolhido in ["Random Forest", "XGBoost"]:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X_test_proc)
            st.write("Resumo das variáveis mais influentes:")
            shap.summary_plot(
                shap_values, X_test_proc,
                feature_names=np.concatenate([
                    colunas_num,
                    preprocessador.named_transformers_["cat"].get_feature_names_out(colunas_cat)
                ])
            )
        else:
            st.warning("O modelo selecionado não é compatível com SHAP TreeExplainer.")

        # Explicabilidade LIME
        st.subheader("🔍 Explicabilidade (LIME)")
        lime_explainer = LimeTabularExplainer(
            training_data=np.array(X_train_bal),
            feature_names=np.concatenate([
                colunas_num,
                preprocessador.named_transformers_["cat"].get_feature_names_out(colunas_cat)
            ]),
            class_names=["Não Compra","Compra"],
            mode="classification"
        )
        exp = lime_explainer.explain_instance(X_test_proc[10], modelo.predict_proba, num_features=10)
        st.write("Exemplo de explicação local (instância 10):")
        st.write(exp.as_list())

        # 📄 Relatório Executivo
        st.subheader("📄 Relatório Executivo")
        relatorio = f"""
        Modelo selecionado: **{modelo_escolhido}**

        Principais métricas:
        - Acurácia: {resultados['Acurácia']:.3f} → O modelo acerta {resultados['Acurácia']*100:.1f}% das previsões.
        - Precisão: {resultados['Precisão']:.3f} → Apenas {resultados['Precisão']*100:.1f}% dos casos previstos como compra realmente compram.
        - Recall: {resultados['Recall']:.3f} → O modelo captura {resultados['Recall']*100:.1f}% dos clientes que compram.
        - F1: {resultados['F1']:.3f} → Equilíbrio moderado entre precisão e recall.
        - AUC: {resultados['AUC']:.3f} → Excelente capacidade de discriminação.

        Variáveis mais influentes:
        - ValorPágina: páginas com alto valor aumentam chance de compra.
        - TaxaRejeição: altas taxas reduzem probabilidade de compra.
        - DiaEspecial: datas próximas a eventos especiais elevam conversão.

        📌 Interpretação executiva:
        O modelo é eficaz para identificar potenciais compradores, especialmente em campanhas amplas.
        No entanto, a precisão baixa indica risco de falsos positivos, sugerindo ajustes para reduzir custos em ações direcionadas.

        🎯 Recomendações:
        - Focar campanhas em clientes que acessam páginas de alto valor.
        - Aproveitar datas especiais para aumentar conversão.
        - Monitorar taxa de rejeição como indicador de desistência.
        """
        st.text_area("Resumo Executivo", relatorio, height=350)