#BIAS
#Imagine uma regressao linear, essa regressao linear nao tem a habilidade replicar o arco da relacao verdadeira
#Logo ela nunca vai ser capaz de encaixar perfeitamente a linha
#A inabilidade de nao captar a verdadeira relacao entre as variaveis e chamado de bias
#Interessantemente, se voce montar uma linha que se encaixa perfeitamente ao testing set, ela provalvemente nao vai se encaixar bem
#No testing set, caso voce a avaliar pela soma dos valores ao quadrado(para evitar distancias negativas)
#VARIANCIA
#Ae que entra a variancia, a variancia seria a soma dos valores ao quadrado para cada dataset
#Caso essa distancia seja alta, significa que esta sofrendo algo chamado overfitting
#Logo, a linha que encaixa perfeitamente pode executar bem mas tbm pode executar muito mau, em prooximo testes,
#por isso o nome variancia
#Agora, a linha reta ela tem um high bias pq ela nao encaixa perfeitamente, mas uma variancia baixa porque ela encaixa bem no testing set
#O seja ela consistemente vai dar boas predicoes, nao perfeitas mas boas
#O algoritmo, perfeito tem low bias e low variancia
#Existe multiplos metodos de misturas essas o sweet spot entre essa duas funcoes, 
#os metodos usados seriam regularization, boosting or bagging
