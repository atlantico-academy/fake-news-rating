# importando bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk import tokenize
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

staff=[
    {
        'nome': 'Carlos Matheus R. Martins', 
        'bio': 'Estudante de Engenharia de software e computação cognitiva.',
        'linkedin': '[Carlos Matheus](https://github.com/cmatheusIA)',
        'github': '[Carlos Matheus](https://www.linkedin.com/in/carlos-matheus-dev/)',
        'email': 'cmatheusrm@alu.ufc.br',
        'imagem': '../docs/assets/images/matheus.png'
    },
    {
        'nome': 'Cristina Toshie Iwassaki', 
        'bio': 'Formada em Física médica pela Unesp de Botucatu. Estudo desenvolvimento web.',
        'linkedin': '[Cristina Iwassaki](https://www.linkedin.com/in/cristina-iwassaki/)',
        'github': '[Cristina Iwassaki](https://github.com/c-Tos1wa)',
        'email': 'cristoshiwassaki@gmail.com',
        'imagem': '../docs/assets/images/profileCristina.jpeg'
    },
    {
        'nome': 'Douglas da Silva Teixeira', 
        'bio': 'Estudante de Física na UFC e de Análise e Desenvolvimento de Sistemas',
        'linkedin': '[Douglas Teixeira](https://www.linkedin.com/in/douglas-teixeira-6854581aa/)',
        'github': '[Douglas Teixeira](https://github.com/DougTeixeira)',
        'email': 'dougteixeira@hotmail.com',
        'imagem': '../docs/assets/images/me.jpg'
    },
    {
        'nome': 'Fco Rafael de L. Xavier', 
        'bio': 'Formado em Oceanografia pela UFC, Mestre em Ciências Marinhas tropicais pela UFC e doutorando na UFC.',
        'linkedin': '[Rafael Xavier](https://www.linkedin.com/in/rafaellxavier)',
        'github': '[Rafael Xavier](https://github.com/rafaelxavier-ocn)',
        'email': 'frlxavier02@gmail.com',
        'imagem': '../docs/assets/images/rafael.jpeg'
    }
]

# carregando modelo
pipe_lr = joblib.load(
    open(
        "../models/model_RandomForestClassifier_with_stop_words_stemma.joblib",
        "rb"
    )
)

# carregando pre processador
pipe_pp = joblib.load(
    open(
        "../models/preprocessor.joblib",
        "rb"
    )
)


nltk.download('rslp')
token_space = tokenize.WhitespaceTokenizer()

# funções
class Processing():
    
    # retirando pontuação
    def clean_dots(self, df):
        df["texto"]=df.texto.apply(lambda x : x.lower())
        df["texto"]=df.texto.apply(lambda x : re.sub('[^\w\s]', '', x))
        return df
   
    # fazendo stemmatização
    def stemma(self, df):
        stemmer = nltk.stem.RSLPStemmer()
        frase_processada = list()
        for text in df["texto"]:
            #para cada frase acessada vamos tokeniza-la e verificar se cada token 
            #é pertecentes a nossa lista de stop
            #words
            nova_frase = list()
            palavras_texto = token_space.tokenize(text)
            for palavra in palavras_texto:
                nova_frase.append(stemmer.stem(palavra)) #stemmatização de cada palavra da sentença
            frase_processada.append(' '.join(nova_frase))
        
        df["text"]=frase_processada
        return df

    
    # Criando bag of words
    def vetorizing(self, df):
        # st.write(pipe_pp)
        return pipe_pp.transform(df["text"])
        
# predição
def predict_model(docx):
    results = pipe_lr.predict(docx)
    return results[0]

# probabilidade
def get_predictiton_proba(docx):
    results = pipe_lr.predict_proba(docx)
    return results

processing = Processing()

def main():
    # titulo
    # Menu
    selected = option_menu(
        menu_title=None,
        options=["Home", "Projeto", "Sobre"],
        icons=["house", "book", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "RebeccaPurple", "font-weight": "normal"},
    }
    )
    
    if selected=="Home":
        st.title("Classificador de notícias") # fake news predictor
    # menu = ['Política', 'Tv e celebridades', 'Sociedade e cotidiano',
    #             'Ciência e tecnologia', 'Economia', 'Religião']
    
    # colocar um help em cada função
        with st.form(key='fake_form'):
    # choice = st.selectbox("Categoria", menu)
            raw_text = st.text_area("Insira o texto aqui","""Dr. Ray peita Bolsonaro, chama-o de conservador fake em entrevista a Danilo Gentili e divide a direita. Este site vem avisando Jair Bolsonaro que ele deveria abandonar a pauta estatista de vez e fazer um discurso mais convincente para aquela boa parte dos liberais e conservadores do Brasil que querem se ver livres das amarras estatais. Tudo bem que as pesquisas ainda dizem que a maior parte do povo é contra as privatizações, mas o índice (pouco mais de 50% do povo) é fácil de ser revertido. Ademais, Bolsonaro deveria falar para direitistas em vez de focar tanto em petistas arrependidos. Recentemente ele disse que pensaria 200 vezes antes de privatizar a Petrobrás para que ela não caia nas mãos de chineses (ou algo do tipo). Deveria ter dito: Eu garanto a privatização da Petrobrás, e também garanto que chineses não irão comprá-la. Isso não deixaria brechas. Do jeito que ele falou, parece que o suposto medo de venda aos chineses é pretexto para evitar a privatização. Seja lá como for, a direita vai ter que adotar alternativas que foquem em um estado reduzido, diminuição de impostos e venda de estatais. Além de João Amoedo, Dr. Rey está fazendo vicejar este tipo de discurso e ainda que sua candidatura esteja em fase inicial é complicado para Bolsonaro que apareçam pessoas de direita propondo uma visão economicamente direitista para a economia. Enfim, veja aos 32:40 Dr. Rey espinafrando Bolsonaro: Quem dá brechas não pode reclamar que os outros aproveitem, não é mesmo?""")
            submit_text = st.form_submit_button(label='Submit')
 
        
        if submit_text:
            df = pd.DataFrame({"texto": [raw_text]})
            
            # processando o texto
            df = processing.clean_dots(df)
            df = processing.stemma(df)
            bag = processing.vetorizing(df)
        
        
            # Aplicando as funções
            prediction = predict_model(bag)
            probability = get_predictiton_proba(bag)
        
            if prediction == 1:
                if np.max(probability)>0.6:
                    st.success("Esta noticia tem uma grande probabilidade de ser verdadeira.")
                else:
                    st.warning("Resultado inconclusivo.")
                
                prediction_result = 'verdadeira'
            else:
                if np.max(probability)>0.6:
                    st.error("Cuidado! Esta noticia tem uma grande probabilidade de ser falsa.")
                else:
                    st.warning("Resultado inconclusivo.")
                
                prediction_result = 'falsa'
                    
        
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = np.max(probability)*100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Probabilidade de ser {prediction_result}", 'font': {'size': 24}},
                number = {'suffix': '%'},
                # delta = {'reference': 40, 'increasing': {'color': "RebeccaPurple"}, 'valueformat': to format the number},
                gauge = {
                    'axis': {'range': [None, 100], 'dtick': 20 ,'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue" if prediction else "red"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#ffebcc'},
                        {'range': [40, 60], 'color': '#ffa31a'},
                        {'range': [60, 100], 'color': '#e68a00'}],
                    # 'threshold': {
                    #     'line': {'color': "red", 'width': 1.5},
                    #     'thickness': 1,
                    #     'value': 60}
                }))
            
       
            fig.update_layout(font = {'color': "darkblue", 'family': "Arial"})
        # fig.update_traces(name=200})
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Mais informações sobre a predição"):
                st.markdown("""
                    - 0% - 40%: Resultado pode estar incorreto.
                    - 40% - 60%: Resultado inconclusivo.
                    - 60% - 100%: Resultado pode estar correto.""")
        
        
            

            st.title("Aviso")
            st.write("""Este é apenas um modelo que faz uma previsão utilizando um
            algoritimo de aprendizagegm de máquina. Aconselhamos que acesse sites de
            checagem de notícias como:""")
            col1, col2, col3 = st.columns(3)
        
            
            col1.markdown((
                "- [Boatos](https://www.boatos.org/)\n"
                "- [Aos fatos](https://www.aosfatos.org/)"
            ))
        
          
            col2.markdown((
                "- [Lupa](https://lupa.uol.com.br/)"
                "- [UOL Confere](https://noticias.uol.com.br/confere/)"))


            col3.markdown((
                "- [Fato ou Fake](https://g1.globo.com/fato-ou-fake/)"
                "- [Estadão verifica](https://politica.estadao.com.br/blogs/estadao-verifica/)"
            ))
            
            st.write("Ou outras entidades especializadas em checagem de fatos.")
        
    if selected=="Projeto":
        # st.title(f"{selected} foi selecionado, mostrar infográfico")
        st.image('../docs/assets/images/resumo_grafico.png')
        
    if selected=="Sobre":
        st.header("")
        st.markdown("""<div style="text-align: justify;">Este projeto faz parte do trabalho final do Bootcamp
        em ciência de dados promovido pelo Instituto Atlântico,
        consistindo no desenvolvimento de um modelo de aprendizagem de máquina,
        que classifica e indica a confiabilidade de uma notícia. Para criação do modelo foi utilizada
        uma base de dados com 7200 textos, divididos igualmente entre verdadeiros e falsos.
        Foram analisados diversos modelos para processamento de texto, sendo escolhido o
        melhor dentre eles para gerar o modelo final.</div>""", unsafe_allow_html=True)
        ''
        st.markdown("""<div style="text-align: justify;">Para dúvidas, problemas ou sugestões, entre em contato pelo e-mail dos membros da equipe.</div>""", unsafe_allow_html=True)
        st.header("Equipe")
        
         
        for member in staff:
            col1, col2 = st.columns([0.3,0.7])
            col1.image(member['imagem'])
            col2.subheader(member['nome'])
            col2.markdown(f"""
            {member['bio']}
            
             - {member['github']}
             - {member['email']}
             - {member['linkedin']}
            """)

            

if __name__ == '__main__':
    main()