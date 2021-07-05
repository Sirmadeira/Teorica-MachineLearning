#Aqui eu vou por todos os optimizers
#RIDGE REGRESSION
#Lembrete regularization e um jeito de evitar overfitting e encontra uma funcao com low bias e variancia
#Imagina que a gente pega uma linha e ve qual o melhor encaixe  ao ver o minimo soma de residuos ao quadrado,
#basicamente uma regressao linear
#Imagine que a gente tem so dois pontos, isso significa q o minimo da soma de residuos ao quadrado, e 0
#Isso significa overfit e alta variancia, quando acrescentar mais datasets a serem predito
#A ideia principal, de ride regression e achar uma linha que nao encaixa tao perfeitamente
#Em outras palavras, a gente poim um pouquinho de bias na linha de treinamento mas em retorno a gente diminui a variancia
#Provendo retornos melhores
#Quando a ridge regression minimiza, a soma dos residuos ao quadrado ela acrescenta lambda * a inclinacao ao quadrado
#Essa parte da equacao acrescenta uma penalidade ao metodo tradicional de reducao do residuos ao quadado e o lambda determina
#O quao forte essa penalidade e
#Ao calcularmos a formulinha entre a linha de ridge regression e a linha do least squareds, a gente escolheria a ridge regression
#Pq o valor dela sr**2+lambda*inclinacao**2, e menor
#A intuicao teorica de ridge regression e bem simples,quando a incinacao de uma linha e alta ela vai ser sensivel a pequenas mudancas no eixo x
#Se ela tiver uma baixa inclinacao ela vai ser menos sensivel
#Como a gente decide qual valor dar a lambda?
#A gente se utiliza de cross validation, geralmente 10 fold validation e determina qual tem a menor variancia
#Note que na regressao logistica, nao e a soma dos residuos ao quadrado na formula
#E a soma das likelihoods+lambda*inclinacao**2
#No geral, todos os parametros entram. Ou seja se tiver mais parametros alem da inclinacao, como diferenca de dieta entre ratos etc
#Eles tambem sao multiplicado e elevado aos quadrado menos o y intercept. Porque todos os parametros sao escalanados pelas medidas menos o y intercept
#COISA MAGICA DE RIDGE REGRESSION
#Imagine que a gente tem 10000 mil parametros, nao daria para a gente calcular isso porque seria muito pesado
#Interessantemente, isso e muito comun em genetica
#No entanto com ridge regression penalty, a gente consegue calcular isso com apenas 500 mas isso e so pra depois 
#LASSO REGRESSION
#Muito semelhante a ridge, mas tem algumas diferencas
#A formula e quase a mesma tbm
#Soma dos residuos ao quadrado+lambda * a inclinacao absuluta(nao inclui negativos)
#A grande diferenca entre a ridge e lasso, e que lasso consegue diminuir a inclinacao ate 0 a ridge nao
#E imagine, uma funcao que considere multiplas variaveis. Mas algumas delas sao inuteis, na ridge o valor delas nunca ira ser nulificado
#O que faz dela mais pesado, no entanto na lasso ela vai ser avaliada ate o valor 0. Logo quando as variaveis sao inuteis, e vc quer elimina-las da equacao final
#Lasso e melhor pq nulifica, no entanto quando a maioria das variaveis tem impacto ridge e melhor
#ELASTIC NET REGRESSION
#E a fusaooo das duas antecessoras
#Muito utilizada em deep learning
#Entao a formula e igual a soma dos residuos quadrados + lambda1(lasso) *inclinacao absulata+lambda2(ridge) *inclinacao **2
#No caso mais simples e e so isso
