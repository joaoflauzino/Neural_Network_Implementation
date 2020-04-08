# Bibliotecas necessárias para tratamento dos dados e rede neural
import os
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
from nltk import tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import time
from sklearn.model_selection import train_test_split
stemmer = LancasterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

os.chdir(path)

#Lendo os dados e guardando em um dataframe
df = pd.read_csv('amazon_cells_labelled.csv', sep = ';')
#Separando em base de teste e treino (20% para teste e 80% para treino)
treino, teste = train_test_split(df, test_size=0.2)

#Coletando stopwords em inglês
linguagem = "english"
palavras_ = stopwords.words(linguagem)
stop_words = []
for elemento in palavras_:
    x = elemento.lower()
    stop_words.append(x)

#Organizando dataset para aplicação da rede neural
sentencas_ = treino['OPINIONS'] #Guardando em uma variável
sentencas_ = list(sentencas_) #Transformando em uma lista
classes_ = treino['SENTIMENT'] #Guardando em uma variável
classes_ = list(classes_) #Transformando em uma lista

classes = [] #Vetor para as classes distintas contidas na base
documentos = [] #Documento com as sentenças
palavras = [] #Palavras contidas no dataset

for elemento in range(len(sentencas_)):
    resultado = ''.join([j for j in sentencas_[elemento] if not j.isdigit()]) #Tirar números
    resultado = re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",
                       (resultado)) #Tirar pontuação, caracteres especiais etc
    w = nltk.word_tokenize(resultado) #Tokenizando sentenças
    w = [stemmer.stem(j.lower()) for j in w]
    w = [j for j in w if j not in stop_words]
    palavras.extend(w) #Adicionando sentenças tokenizadas no vetor de palavras
    documentos.append((w, classes_[elemento])) #Adicionando sentenças tokenizadas e
    #classes em documents
    if classes_[elemento] not in classes:
        classes.append(classes_[elemento]) #Montando vetor com as classes distintas


palavras = [stemmer.stem(w.lower()) for w in palavras] #Reduzindo palavras para sua raiz
palavras = [w for w in palavras if w not in stop_words] #Retirando as stopwords
palavras = list(set(palavras)) #Transformando em lista
classes = list(set(classes)) #transformando em lista

#Criando nossos dados de treino
treinamento = []
saida = []
#Criando um arrya vazio para saída
saida_vazia = [0] * len(classes)
#Dados de treinamento, bag of words para cada sentença
for doc in documentos:
    #Inicializando nosso bag of words
    bag = []
    #Lista de palavras tokenizadas
    palavras_patt = doc[0]
    #stem cada palavra
    palavras_patt = [stemmer.stem(palavra.lower()) for palavra in palavras_patt]
    #criando nosso bag of words
    for w in palavras:
        bag.append(1) if w in palavras_patt else bag.append(0)

    treinamento.append(bag)
    #Saida consiste em 0 e 1
    linha_saida = list(saida_vazia)
    linha_saida[classes.index(doc[1])] = 1
    saida.append(linha_saida)

# sample training/output
i = 3
w = documentos[i][0]
print ("Sentenca tratada: ", w)
print ("Bag of Words: ", treinamento[i])
print ("Classe: ", saida[i])

#Funções para as entradas de teste

#Calculando sigmoid
def sigmoid(x):
    saida = 1/(1+np.exp(-x))
    return saida

#Derivando resultado da sigmoide
def sigmoid_derivada(saida):
    return saida*(1-saida)

#Limpando sentença
def limpando_sentenca(sentenca):
    #Tokenizando sentença
    sentenca_palavras = nltk.word_tokenize(sentenca)
    #Reduzindo palavras para sua raiz
    sentenca_palavras = [stemmer.stem(palavra.lower()) for palavra in sentenca_palavras]
    #Retirando stopwords
    sentenca_palavras = [w for w in sentenca_palavras if w not in stop_words]
    return sentenca_palavras

#Verifica o bag of words: 0 ou 1 para cada palavra contida ou não no bag
def bag(sentenca, palavras, show_details=False):
    #tokeniza a sentença
    sentenca_palavras = limpando_sentenca(sentenca)
    #bag of words
    bag = [0]*len(palavras)
    for s in sentenca_palavras:
        for i,w in enumerate(palavras):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def processamento(sentenca, show_details=False):
    x = bag(sentenca.lower(), palavras, show_details)
    if show_details:
        print ("Sentença:", sentenca, "\n Bag:", x)
    #Input é o bag of words
    l0 = x
    #Multiplicação de matriz da entrada e camada escondida
    l1 = sigmoid(np.dot(l0, synapse_0))
    #Camada de saída
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

def treino(X, y, neuronios_camada_escondida=20, alpha=1, epocas=10000, dropout=False, dropout_porcentagem=0.5):
    print ("Treinamento com %s neuronios, alpha:%s, dropout:%s %s" % (neuronios_camada_escondida, str(alpha), dropout,
    dropout_porcentagem if dropout else '') )
    print ("Matriz de entrada: %sx%s    Matriz de saída: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)
    last_mean_error = 1
    #Inicializando os pesos com a media 0
    synapse_0 = 2*np.random.random((len(X[0]), neuronios_camada_escondida)) - 1
    synapse_1 = 2*np.random.random((neuronios_camada_escondida, len(classes))) - 1
    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
    for j in iter(range(epocas+1)):
        #Avançando pelas camadas 0,1 e 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),neuronios_camada_escondida))],
                                          1-dropout_porcentagem)[0] * (1.0/(1-dropout_porcentagem))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        #Quando foi perdido do valor alvo?
        layer_2_error = y - layer_2
        if (j% 10000) == 0 and j > 5000:
            #Se o erro de 10k iterações for maior do que a iteração interior, parar.
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("Delta após "+str(j)+" Iterações:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("Parada:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
        #A função alvo está indo na direção correta?
        layer_2_delta = layer_2_error * sigmoid_derivada(layer_2)
        #De acordo com os pesos, qando cada valor do l1 está contribuindo para o erro em l2?
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        #Em qual direção está indo l1?
        layer_1_delta = layer_1_error * sigmoid_derivada(layer_1)
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
    now = datetime.datetime.now()
    #Synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'Timestamp': now.strftime("%Y-%m-%d %H:%M"),
               'Palavras': palavras,
               'Classes': classes
              }
    synapse_file = "synapses.json"
    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("Synapses salva em:", synapse_file)


X = np.array(treinamento) #Transformando os dados de treino em um array
y = np.array(saida) #Transformando os dados de saido em um array

inicio = time.time() #Tempo

#Treinando a rede neural
treino(X, y, neuronios_camada_escondida=20, alpha=0.1, epocas=100000, dropout=False, dropout_porcentagem=0.2)

tempo_gasto = time.time() - inicio #Tempo
print ("Tempo de processamento:", tempo_gasto/60, "minutos")

#Corte da probabilidade, considerando todas classificacoes acima de 0.2
ERRO_THRESHOLD = 0.2
#Consultando valores da synapse
arq_synapse = 'synapses.json'
with open(arq_synapse) as dados_arq:
    synapse = json.load(dados_arq)
    synapse_0 = np.asarray(synapse['synapse0'])
    synapse_1 = np.asarray(synapse['synapse1'])

#Classificando sentença
def classificacao(sentenca, show_details=False):
    resultados = processamento(sentenca, show_details)
    resultados = [[i,r] for i,r in enumerate(resultados) if r>ERRO_THRESHOLD]
    resultados.sort(key=lambda x: x[1], reverse=True)
    resultados_finais =[[classes[r[0]],r[1]] for r in resultados]
    return resultados_finais


if __name__ == "__main__":

    exemplos_negativos = teste[teste['SENTIMENT'] == 0] #Dados de teste com comentarios negativos
    exemplos_positivos = teste[teste['SENTIMENT'] == 1] #Dados de teste com comentarios negativos


    #Contando quantas classificações negativas aconteceram
    qtd_negativos_corretos = 0
    qtd_negativos_errados = 0

    print('-------------NEGATIVAS-------------\n')
    for elemento in range(len(list(exemplos_negativos['OPINIONS']))):
        resultados = []
        resultados_finais = []
        x = classificacao(list(exemplos_negativos['OPINIONS'])[elemento])
        for i in range(len(x)):
            resultados.append(i)
            resultados.append(x[i][0])
            resultados.append(x[i][1])
            if x[i][0] == 1:
                qtd_negativos_errados += 1
            if x[i][0] == 0:
                qtd_negativos_corretos += 1
        resultados_finais.append(resultados)

    print('Negativos Falsos: ', qtd_negativos_errados)
    print('Negativos Corretos: ', qtd_negativos_corretos)
    print('Acertou Negativos%: ', (qtd_negativos_corretos/(qtd_negativos_errados + qtd_negativos_corretos))*100)


    #Contando quantas classificações positivas aconteceram
    qtd_positivos_corretos = 0
    qtd_positivos_errados = 0

    print('\n-------------POSITIVAS-------------\n')
    for elemento in range(len(list(exemplos_positivos['OPINIONS']))):
        resultados = []
        resultados_finais = []
        x = classificacao(list(exemplos_positivos['OPINIONS'])[elemento])
        for i in range(len(x)):
            resultados.append(i)
            resultados.append(x[i][0])
            resultados.append(x[i][1])
            if x[i][0] == 1:
                qtd_positivos_corretos += 1
            if x[i][0] == 0:
                qtd_positivos_errados += 1
        resultados_finais.append(resultados)


    print('Positivos Falsos: ', qtd_positivos_errados)
    print('Positivos Corretos: ', qtd_positivos_corretos)
    print('Acertou Positivos%: ', (qtd_positivos_corretos/(qtd_positivos_corretos + qtd_positivos_errados))*100)

    print('\n-------------ACURÁCIA-------------\n')

    print('Acertou no Total%: ', ((qtd_positivos_corretos + qtd_negativos_corretos)/(qtd_positivos_corretos + qtd_positivos_errados + qtd_negativos_errados + qtd_negativos_corretos))*100)
