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
        "nav-link-selected": {"background-color": "RebeccaPurple"},
    }
    )
    
    if selected=="Home":
        st.title("Classificador de notícias") # fake news predictor
    # menu = ['Política', 'Tv e celebridades', 'Sociedade e cotidiano',
    #             'Ciência e tecnologia', 'Economia', 'Religião']
    
    # colocar um help em cada função
        with st.form(key='fake_form'):
    # choice = st.selectbox("Categoria", menu)
            raw_text = st.text_area("insira o texto aqui","""Dr. Ray peita Bolsonaro, chama-o de conservador fake em entrevista a Danilo Gentili e divide a direita. Este site vem avisando Jair Bolsonaro que ele deveria abandonar a pauta estatista de vez e fazer um discurso mais convincente para aquela boa parte dos liberais e conservadores do Brasil que querem se ver livres das amarras estatais. Tudo bem que as pesquisas ainda dizem que a maior parte do povo é contra as privatizações, mas o índice (pouco mais de 50% do povo) é fácil de ser revertido. Ademais, Bolsonaro deveria falar para direitistas em vez de focar tanto em petistas arrependidos. Recentemente ele disse que pensaria 200 vezes antes de privatizar a Petrobrás para que ela não caia nas mãos de chineses (ou algo do tipo). Deveria ter dito: Eu garanto a privatização da Petrobrás, e também garanto que chineses não irão comprá-la. Isso não deixaria brechas. Do jeito que ele falou, parece que o suposto medo de venda aos chineses é pretexto para evitar a privatização. Seja lá como for, a direita vai ter que adotar alternativas que foquem em um estado reduzido, diminuição de impostos e venda de estatais. Além de João Amoedo, Dr. Rey está fazendo vicejar este tipo de discurso e ainda que sua candidatura esteja em fase inicial é complicado para Bolsonaro que apareçam pessoas de direita propondo uma visão economicamente direitista para a economia. Enfim, veja aos 32:40 Dr. Rey espinafrando Bolsonaro: Quem dá brechas não pode reclamar que os outros aproveitem, não é mesmo?""")
            submit_text = st.form_submit_button(label='submit')
 
        
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
                st.success("Esta noticia pode ser verdadeira.")
            else:
                if np.max(probability)>0.6:
                    st.error("Atenção! Esta noticia pode ser falsa.")
                else:
                    st.warning("Atenção! Esta noticia pode ser falsa.")
        
            st.markdown(f"- 0% - 40%: Resultado pode estar incorreto.")
            st.markdown(f"- 40% - 60%: Resultado inconclusivo.")
            st.markdown(f"- 60% - 100%: Resultado pode estar correto.")

            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = np.max(probability)*100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilidade (%)", 'font': {'size': 24}},
                # delta = {'reference': 40, 'increasing': {'color': "RebeccaPurple"}},
                gauge = {
                    'axis': {'range': [None, 100], 'dtick': 20 ,'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#ffebcc'},
                        {'range': [40, 60], 'color': '#ffa31a'},
                        {'range': [60, 100], 'color': '#e68a00'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 1.5},
                        'thickness': 1,
                        'value': 60}
                }))
            
       
            fig.update_layout(font = {'color': "darkblue", 'family': "Arial"})
        # fig.update_traces(name=200})
            st.plotly_chart(fig, use_container_width=True)
        
        
            

            st.title(f"Aviso")
            st.write(f"""Este é apenas um modelo que faz uma previsão utilizando um
            algoritimo de aprendizagegm de máquina. Aconselhamos que acesse sites de
            checagem de notícias como:""")
            col1, col2, col3 = st.columns(3)
        
            with col1:
                st.markdown(f"- [Boatos](https://www.boatos.org/)")
                st.markdown(f"- [Aos fatos](https://www.aosfatos.org/)")
        
            with col2:
                st.markdown(f"- [Lupa](https://lupa.uol.com.br/)")
                st.markdown(f"- [UOL Confere](https://noticias.uol.com.br/confere/)")
        
            with col3:
                st.markdown(f"- [Fato ou Fake](https://g1.globo.com/fato-ou-fake/)")
                st.markdown(f"- [Estadão verifica](https://politica.estadao.com.br/blogs/estadao-verifica/)")
            
            st.write("Ou outras entidades especializadas em checagem de fatos.")
        
    if selected=="Projeto":
        # st.title(f"{selected} foi selecionado, mostrar infográfico")
        st.image('../docs/assets/images/resumo_grafico.png')
        
    if selected=="Sobre":
        st.header(f"{selected}")
        st.markdown("""<div style="text-align: justify;">Este projeto faz parte do trabalho final do Bootcamp em ciência de dados promovido pelo Instituto Atlântico,
        consistindo no desenvolvimento de um modelo de aprendizagem de máquina,
        que classifica e indica a confiabilidade de uma notícia. Para criação do modelo foi utilizada
        uma base de dados com 7200 textos, divididos igualmente entre verdadeiros e falsos.
        Foram analisados diversos modelos para processamento de texto, sendo escolhido o
        melhor dentre eles para gerar o modelo final.</div>""", unsafe_allow_html=True)
        ''
        st.markdown("""<div style="text-align: justify;">Para dúvidas, problemas ou sugestões, entre em contato pelo e-mail dos membros da equipe.</div>""", unsafe_allow_html=True)
        st.header("Equipe")
        col1, col2 = st.columns(2)
        with col1:
            'foto?'
            # st.image()
            # st.image()
            # st.image()
        
        with col2:
            st.subheader("Amanda da Silva Farias")
            st.markdown("""<div style="text-align: justify;">Cientista da Computação
            com especialização em Engenharia de Software.</div>""", unsafe_allow_html=True)
            ''
            st.write(f"e-mail: amandafharias@gmail.com")
            st.markdown(f"github: [Amanda Farias](https://github.com/AmandaFar/)")
            # st.markdown(f"Linkedin: [Amanda Farias]()")
            ''
            st.subheader("Carlos Matheus Rodrigues Martins")
            st.markdown("""<div style="text-align: justify;">Estudante de Engenharia
            de software e computação cognitiva.</div>""", unsafe_allow_html=True)
            ''
            st.markdown(f"e-mail: cmatheusrm@alu.ufc.br")
            st.markdown(f"github: [Carlos Matheus](https://github.com/cmatheusIA)")
            st.markdown(f"Linkedin: [Carlos Matheus](https://www.linkedin.com/in/carlos-matheus-dev/)")
            ''
            st.subheader("Cristina Toshie Iwassaki")
            st.markdown("""<div style="text-align: justify;">Formada em Física médica
            pela Unesp de Botucatu. Estudo desenvolvimento web.</div>""", unsafe_allow_html=True)
            ''
            st.markdown(f"e-mail: cristoshiwassaki@gmail.com")
            st.markdown(f"github: [Cristina Iwassaki](https://github.com/c-Tos1wa)")
            st.markdown(f"Linkedin: [Cristina Iwassaki](https://www.linkedin.com/in/cristina-iwassaki/)")
            ''
            st.subheader("Douglas da Silva Teixeira")
            st.markdown("""<div style="text-align: justify;">Estudante de Física na
            UFC e de Análise e Desenvolvimento de Sistemas.</div>""", unsafe_allow_html=True)
            ''
            st.markdown(f"e-mail: dougteixeira@hotmail.com")
            st.markdown(f"github: [Douglas Teixeira](https://github.com/DougTeixeira)")
            st.markdown(f"Linkedin: [Douglas Teixeira](https://www.linkedin.com/in/douglas-teixeira-6854581aa/)")
            ''
            st.subheader("Francisco Rafael de Lima Xavier")
            st.markdown("""<div style="text-align: justify;">Formado em Oceanografia
            pela UFC, Mestre em Ciências Marinhas tropicais pela UFC e doutorando na UFC.</div>""", unsafe_allow_html=True)
            ''
            st.markdown(f"e-mail: frlxavier02@gmail.com")
            st.markdown(f"github: [Rafael Xavier](https://github.com/rafaelxavier-ocn)")
            st.markdown(f"Linkedin: [Rafael Xavier](https://www.linkedin.com/in/rafaellxavier)")
            ''
            st.subheader("Hosana Fernandes Gomes")
            st.markdown("""<div style="text-align: justify;">Arquiteta e urbanista pela UFC,
            Mestrado em Design da Informação pelo PPGAUD-UFC.</div>""", unsafe_allow_html=True)
            ''
            st.markdown(f"e-mail: fernandeshosana48@gmail.com")
            st.markdown(f"github: [Hosana Fernandes](http://github.com/hosanafg)")
            st.markdown(f"Linkedin: [Hosana Fernandes](https://www.linkedin.com/in/hosana-fernandes-772b94107/)")
            

if __name__ == '__main__':
    main()