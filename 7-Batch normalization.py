#Normalization e quando voce executa a seguinte formula, z=x-m/s, m e a media do dataset, s e o standard deviation e o x e o data point
#A primeira coisa que acontece numa batch normalization, e normalizar o output de uma funcao de ativacao,
# ou seja executar a funcao acima sendo x o ponto de interesee
#Depois ele multiplica o output por um parametro arbitrario z*g
#E por fim, ele soma outra parametro arbitrario(x*g)+b
#Ele aumenta a velocidade de analise de dados, e alem do mais corrige pesos outliers