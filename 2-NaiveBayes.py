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