# Imports
import pandas as pd
import streamlit as st
import numpy as np

from datetime import datetime
from PIL import Image
from io import BytesIO

# Importações adicionais para K-Means e visualização
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def convert_df(df):
    """Converte um DataFrame para CSV."""
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def to_excel(df):
    """Converte um DataFrame para Excel."""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

### Criando os segmentos
def recencia_class(x, r, q_dict):
    """
    Classifica a recência. O menor quartil é considerado o melhor ('A').
    x = valor da linha,
    r = recencia,
    q_dict = dicionário de quartis
    """
    if x <= q_dict[r][0.25]:
        return 'A'
    elif x <= q_dict[r][0.50]:
        return 'B'
    elif x <= q_dict[r][0.75]:
        return 'C'
    else:
        return 'D'

def freq_val_class(x, fv, q_dict):
    """
    Classifica frequência ou valor. O maior quartil é considerado o melhor ('A').
    x = valor da linha,
    fv = frequencia ou valor,
    q_dict = dicionário de quartis
    """
    if x <= q_dict[fv][0.25]:
        return 'D'
    elif x <= q_dict[fv][0.50]:
        return 'C'
    elif x <= q_dict[fv][0.75]:
        return 'B'
    else:
        return 'A'

# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title='RFV e K-Means', layout="wide", initial_sidebar_state='expanded')

    # Título principal da aplicação
    st.write("""# RFV e K-Means para Segmentação de Clientes

    RFV significa Recência, Frequência, Valor e é utilizado para segmentação de clientes baseado no comportamento
    de compras dos clientes, agrupando-os em clusters parecidos. Utilizando esse tipo de agrupamento podemos realizar
    ações de marketing e CRM melhor direcionadas, ajudando assim na personalização do conteúdo e até a retenção de clientes.

    Para cada cliente é preciso calcular cada uma das componentes abaixo:

    - **Recência (R):** Quantidade de dias desde a última compra.
    - **Frequência (F):** Quantidade total de compras no período.
    - **Valor (V):** Total de dinheiro gasto nas compras do período.

    Além da segmentação por quartis, exploraremos a segmentação por K-Means para uma clusterização mais completa.
    """)
    st.markdown("---")

    # Apresenta a imagem na barra lateral da aplicação (descomente para usar)
    # image = Image.open("Bank-Branding.jpg")
    # st.sidebar.image(image)

    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Carregue seus dados de compras (CSV ou XLSX)", type=['csv', 'xlsx'])

    # Verifica se há conteúdo carregado na aplicação
    if data_file_1 is not None:
        if data_file_1.name.endswith('.csv'):
            df_compras = pd.read_csv(data_file_1, infer_datetime_format=True, parse_dates=['DiaCompra'])
        else:
            df_compras = pd.read_excel(data_file_1, infer_datetime_format=True, parse_dates=['DiaCompra'])

        st.write('## 1. Cálculo das Métricas RFV')

        st.write('### Recência (R)')
        dia_atual = df_compras['DiaCompra'].max()
        st.write(f'Dia máximo na base de dados: **{dia_atual.strftime("%d/%m/%Y")}**')
        st.write('Calculando quantos dias se passaram desde a última compra de cada cliente.')

        df_recencia = df_compras.groupby(by='ID_cliente', as_index=False)['DiaCompra'].max()
        df_recencia.columns = ['ID_cliente', 'DiaUltimaCompra']
        df_recencia['Recencia'] = df_recencia['DiaUltimaCompra'].apply(lambda x: (dia_atual - x).days)
        st.write(df_recencia.head())
        df_recencia.drop('DiaUltimaCompra', axis=1, inplace=True)

        st.write('### Frequência (F)')
        st.write('Calculando quantas vezes cada cliente comprou conosco.')
        df_frequencia = df_compras[['ID_cliente', 'CodigoCompra']].groupby('ID_cliente').count().reset_index()
        df_frequencia.columns = ['ID_cliente', 'Frequencia']
        st.write(df_frequencia.head())

        st.write('### Valor (V)')
        st.write('Calculando o total de dinheiro gasto por cada cliente no período.')
        df_valor = df_compras[['ID_cliente', 'ValorTotal']].groupby('ID_cliente').sum().reset_index()
        df_valor.columns = ['ID_cliente', 'Valor']
        st.write(df_valor.head())

        st.write('### Tabela RFV Final')
        df_RF = df_recencia.merge(df_frequencia, on='ID_cliente')
        df_RFV = df_RF.merge(df_valor, on='ID_cliente')
        df_RFV.set_index('ID_cliente', inplace=True)
        st.write(df_RFV.head())

        st.write('## 2. Segmentação por Quartis (RFV Score)')
        st.write("Uma forma de segmentar os clientes é criando quartis para cada componente do RFV. O melhor quartil é chamado de 'A', o segundo melhor de 'B', o terceiro de 'C' e o pior de 'D'. A definição de 'melhor' ou 'pior' depende da componente:")
        st.markdown("""
        - **Recência:** Quanto menor a recência, melhor (menor quartil = 'A').
        - **Frequência:** Quanto maior a frequência, melhor (maior quartil = 'A').
        - **Valor:** Quanto maior o valor gasto, melhor (maior quartil = 'A').
        """)
        st.write('Se desejar mais ou menos classes, basta ajustar o número de quantis.')

        st.write('### Quartis para o RFV')
        quartis = df_RFV.quantile(q=[0.25, 0.5, 0.75])
        st.write(quartis)

        st.write('### Tabela após a criação dos grupos por Quartis')
        df_RFV['R_quartil'] = df_RFV['Recencia'].apply(recencia_class, args=('Recencia', quartis))
        df_RFV['F_quartil'] = df_RFV['Frequencia'].apply(freq_val_class, args=('Frequencia', quartis))
        df_RFV['V_quartil'] = df_RFV['Valor'].apply(freq_val_class, args=('Valor', quartis))
        df_RFV['RFV_Score'] = df_RFV['R_quartil'] + df_RFV['F_quartil'] + df_RFV['V_quartil']
        st.write(df_RFV.head())

        st.write('### Quantidade de clientes por RFV Score')
        st.write(df_RFV['RFV_Score'].value_counts().sort_index())

        st.write('#### Exemplo: Clientes com menor recência, maior frequência e maior valor gasto (Score "AAA")')
        st.write(df_RFV[df_RFV['RFV_Score'] == 'AAA'].sort_values('Valor', ascending=False).head(10))

        st.write('## 3. Segmentação por K-Means')
        st.write('Para uma clusterização mais completa e baseada em similaridade, aplicaremos o algoritmo K-Means. Antes de aplicar, é fundamental **escalar os dados** para que todas as variáveis (Recência, Frequência, Valor) tenham a mesma importância na formação dos clusters.')

        # Escalonando os dados
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_RFV[['Recencia', 'Frequencia', 'Valor']])
        df_scaled = pd.DataFrame(df_scaled, columns=['Recencia_scaled', 'Frequencia_scaled', 'Valor_scaled'], index=df_RFV.index)
        st.write('### Dados RFV Escalonados')
        st.write(df_scaled.head())

        st.write('### Método do Cotovelo para Determinar o Número Ideal de Clusters')
        st.write('O método do cotovelo ajuda a encontrar o número ideal de clusters (k). Ele plota a soma dos quadrados dos erros (SSE) para diferentes valores de k. O "cotovelo" no gráfico indica o ponto onde a diminuição da SSE começa a desacelerar, sugerindo um bom equilíbrio entre o número de clusters e a distorção.')

        sse = {}
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init para suprimir warning
            kmeans.fit(df_scaled)
            sse[k] = kmeans.inertia_

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(list(sse.keys()), list(sse.values()), marker='o')
        ax.set_xlabel('Número de Clusters (k)')
        ax.set_ylabel('Soma dos Quadrados dos Erros (SSE)')
        ax.set_title('Método do Cotovelo para K-Means')
        st.pyplot(fig)
        st.write('Observe o gráfico acima para identificar o "cotovelo", que indica o número ideal de clusters para seus dados.')

        # Slider para o usuário selecionar o número de clusters
        n_clusters = st.slider('Selecione o número de clusters para o K-Means:', min_value=2, max_value=10, value=4)

        # Aplicando K-Means com o número de clusters selecionado
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init para suprimir warning
        df_RFV['Cluster_KMeans'] = kmeans.fit_predict(df_scaled)
        st.write(f'### Tabela RFV com os {n_clusters} Clusters K-Means')
        st.write(df_RFV.head())

        st.write('### Análise dos Clusters K-Means')
        st.write('A média das métricas RFV para cada cluster K-Means pode ajudar a entender o perfil de cada grupo de clientes.')
        cluster_analysis = df_RFV.groupby('Cluster_KMeans')[['Recencia', 'Frequencia', 'Valor']].mean()
        st.write(cluster_analysis)

        st.write('### Distribuição de clientes por Cluster K-Means')
        st.write(df_RFV['Cluster_KMeans'].value_counts().sort_index())

        st.write('## 4. Ações de Marketing/CRM')
        st.write('As ações de marketing podem ser direcionadas com base nos segmentos RFV e, agora, também nos clusters K-Means. Abaixo, um exemplo de mapeamento de ações para alguns scores RFV:')

        dict_acoes = {
            'AAA': 'Enviar cupons de desconto, Pedir para indicar nosso produto para algum amigo, Ao lançar um novo produto enviar amostras grátis para esses.',
            'DDD': 'Churn! Clientes que gastaram bem pouco e fizeram poucas compras. Fazer nada ou tentar uma reativação com baixo custo.',
            'DAA': 'Churn! Clientes que gastaram bastante e fizeram muitas compras, mas estão com alta recência. Enviar cupons de desconto para tentar recuperar.',
            'CAA': 'Churn! Clientes que gastaram bastante e fizeram muitas compras, mas estão com alta recência. Enviar cupons de desconto para tentar recuperar.'
        }

        df_RFV['acoes de marketing/crm'] = df_RFV['RFV_Score'].map(dict_acoes)
        st.write(df_RFV.head())

        st.write('### Quantidade de clientes por tipo de ação (baseado no RFV Score)')
        st.write(df_RFV['acoes de marketing/crm'].value_counts(dropna=False))

        # Botão de download do DataFrame RFV completo (agora com K-Means)
        df_xlsx = to_excel(df_RFV)
        st.download_button(label='📥 Baixar Tabela RFV Completa (com K-Means)', data=df_xlsx, file_name='RFV_KMeans_Clientes.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main_':
    main()
