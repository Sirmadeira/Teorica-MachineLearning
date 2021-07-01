#NAIVE BAYES CLASSIFIER
#A galera geralmente chama de multinomial naive bayes classifier
#Exite outro chamado gaussian naive bayes classifier
#Vamo focar no primeiro
#Imagine que voce tem dois tipos de email, spawn e desejados
#Entao a primeira coisa que a gente faz e construir um histograma das palavras que ocorrem no desejados
#E a gente consegue a probabilidade de cada palavra ser achada, nesse email
#A mesma coisa e feita com os de spwan
#O nome dado as probabilidades de palavras, e tbm chamado de likelihood
#Imagine uma pequena mensagem, feita por duas palavras.
#Existe uma prior probability, que ela seja normal baseado no numero de emails normais pelo total(spawn+desejado) de emails.
#Prior probability significa- Probabilidade inicial
#Depois disso a gente multiplica a probabilidade de achar akls palavras no email normal
#E a gente tem a probabilidade, de ser normal e ter akls palavras
#A mesma coisa no spawn
#A probabilidade que for maior sera a probabilidade maior
#Isso e naive bayes classificacao
#Agora imagine uma mensagem que tem a mesma palavra 20 vezes
#Mas ela nunca aparece em spawn,
#Entao ela tem uma probabilidade nula e e julgada como normal, sendo que ela e spawn
#Como consertar?
#Para consertar isso, a gente da uma valor inicial para cada palavra
#isso geralmente e representado por alfa do grego
#Agora agente vai falar pq naive bayes, e inocente
#Ele e considerado inocente pq ele da a mesma probabilidade a palavra, sem considerar a ordem delas
#Entao se eu por em uma ordem nada ver, ele ainda vai considera naive
#Em machine learning, a gente fala que ao ignorar o lingo entre a palavras ele tem um high bias
#Mas ja que ele funfa ele tem uma variancia baixa
#DISTRIBUICAO NORMAL
#Antes de entender o que e uma gaussian naive bayes classifier
#Temos que entender o que e uma distribuicao normal
#Imagine um grafico de altura como eixo x, por numero de pessoas
#O grafico vai ter um centro medio, onde vai ter o maior numero de pessoas. E ele provavlemente vai ser na altura media
#Isso e chamado de uma distribuicao encurvada
#A inclinacao dessa curva e decidida pela deviacao comum, quando tiver uma deviacao baixa
#Ela vai aparentar ter um pino que ascende e cai rapidamente
#Senao e uma mais gradual
#Para desenhar uma distribuicao normal
#Voce precisa saber de duas coisas, a media das medidas. Isso fala onde e o centro do pino
#E a deviacao comun, que como dito acima define a inclinacao da curva
#NAIVE BAYES GAUSSIANO
#Uma curva gaussiana e a mesma coisa que uma distribuicao normal
#Imagine 3 fatores, em relacao a uma pessoa que gosta ou nao de um filme
#Tpo quanto pipoca ela come, quanto doces, e o quanto ela bebe de refrigerante
#Ae a gente desenha, as distribuicoes gaussiana de cada uma dessas estatisticas de acordo com a divisao criada
#Depois disso, a gente multiplica a prior probability (probabilidade inicial)* a likelihood (basicamente ponto no grafico no eixo y, que corresponde ao quanto ela come de pipoca Etc..)
#A mesma coisa e feita com os outros, no entanto um deles teve um valor muitto baixo
#Entao a gente pega o log desse valor, para evitar underflow
#Underflow, ocorre quando o computador nao manten mais o controle do valor muito proximo de 0 ,para evitar erros a gente pega o log de tudo
#Geralmente e o ln (log com base euler),
#E o log transforma a multiplicacao para a soma do logs dos  valores log(prior probability)+log(likelihood de comer pipoca)+log(de bebe soda) e assim vai
#Ae a gente calcula, akl que tiver o maior valor ganha
